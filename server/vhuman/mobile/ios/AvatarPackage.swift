import Foundation
import CryptoKit

enum AvatarError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case let .invalid(reason) = self { return reason }; return nil }
}

struct AvatarPackage {
    struct FileRecord: Decodable { let sha256: String; let bytes: Int }
    struct Manifest: Decodable {
        let schema: String
        let source_geometry_sha256: String
        let files: [String: FileRecord]
    }
    struct Controls: Decodable { let schema: String; let names: [String]; let reference: [Float] }
    let directory: URL
    let geometryHash: String
    let packageHash: String
    let reference: [Float]

    init(directory: URL) throws {
        let decoder = JSONDecoder()
        let manifestData = try Data(contentsOf: directory.appendingPathComponent("avatar.json"))
        let m = try decoder.decode(Manifest.self, from: manifestData)
        guard m.schema == "vhuman.mobile_avatar.v1" else { throw AvatarError.invalid("Unsupported avatar package") }
        let required: Set<String> = ["gnm.bin", "avatar.glb", "streams.bin", "bindings.bin", "controls.json"]
        guard required.isSubset(of: Set(m.files.keys)) else { throw AvatarError.invalid("Incomplete avatar package") }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        for (name, record) in m.files {
            let url = directory.appendingPathComponent(name).resolvingSymlinksInPath().standardizedFileURL
            guard !name.contains("/"), url.path.hasPrefix(root), record.bytes > 0,
                record.bytes <= 256 * 1024 * 1024 else { throw AvatarError.invalid("Invalid avatar file path or size") }
            let input = try FileHandle(forReadingFrom: url)
            defer { try? input.close() }
            var hash = SHA256(); var count = 0
            while let chunk = try input.read(upToCount: 1 << 20), !chunk.isEmpty {
                count += chunk.count
                guard count <= record.bytes else { throw AvatarError.invalid("Avatar size mismatch") }
                hash.update(data: chunk)
            }
            let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == record.sha256, count == record.bytes else { throw AvatarError.invalid("Avatar checksum mismatch: \(name)") }
        }
        let c = try decoder.decode(Controls.self, from: Data(contentsOf: directory.appendingPathComponent("controls.json")))
        guard c.schema == "vhuman.gnm_controls.v1", c.names.count == 383, c.reference.count == 383,
            c.reference.allSatisfy({ $0.isFinite && abs($0) <= 3 }) else { throw AvatarError.invalid("Invalid native control layout") }
        self.directory = directory
        geometryHash = m.source_geometry_sha256
        packageHash = SHA256.hash(data: manifestData).map { String(format: "%02x", $0) }.joined()
        reference = c.reference
    }
}

extension Array where Element == Float {
    var bytes: Data { withUnsafeBytes { Data($0) } }
}
