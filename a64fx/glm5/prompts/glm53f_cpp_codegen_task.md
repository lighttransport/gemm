[gMASK]<sop><|system|>You are a careful C++17 engineer. Return only one complete C++17 source file, with no Markdown fences, no prose, and no analysis. The source must compile with a conforming C++17 compiler using -Wall -Wextra -Werror -pedantic. Do not use nonstandard headers or extensions.<|user|>
Implement a small, standalone C++17 command-line program named task_queue.cpp. It reads commands from standard input and writes exactly one response line per command. This is a coding assessment: correctness, clear invariants, and robust parsing matter more than cleverness.

The program owns a FIFO queue of jobs. A job has an unsigned decimal id in the range 1 through 4294967295 and a payload, which is the remainder of the line after the id. Payloads may contain spaces and may be empty. Lines use Unix newlines. Blank input lines are ignored. Commands are ASCII and case-sensitive:

ADD <id> <payload>
  Insert a new job at the back. If <id> already exists anywhere in the queue, do not modify the queue and print ERR duplicate.
POP
  Remove the oldest job. If the queue is empty print ERR empty. Otherwise print OK <id> <payload>. The output must retain an empty payload as the single trailing space after the id.
PEEK
  Do not remove the oldest job. Its success and error output are identical to POP.
CANCEL <id>
  Remove the job with that id wherever it is. Print OK cancelled <id>, or ERR missing if absent.
COUNT
  Print OK <n>, where n is the current number of jobs.
QUIT
  Print OK bye and stop reading, even if further input exists.

For any malformed command, malformed integer, out-of-range id, missing required argument, or trailing non-whitespace text on a command which takes no arguments, print ERR invalid and leave the queue unchanged. A leading plus sign, a minus sign, hexadecimal notation, decimal fractions, and whitespace-only ids are invalid. Leading zeros are permitted. On ADD, exactly one or more ASCII spaces or tabs must separate ADD, id, and payload; the payload begins after that separator and may be empty. For CANCEL, surrounding ASCII spaces and tabs around the id are allowed, but no other trailing data is allowed. POP, PEEK, COUNT, and QUIT may have trailing spaces or tabs only.

Use std::getline so a payload is never split at whitespace. Preserve payload bytes exactly after its required separator. Use a queue-friendly representation and an id lookup that remains valid after cancellation. Avoid undefined behavior while parsing 32-bit unsigned integers. Important edge case: ID zero is invalid; for example, `ADD 0 zero` must print `ERR invalid` and leave the queue unchanged. Do not define unused helpers: the exact compiler uses `-Werror`, so an unused-function warning is a build failure. Keep the code self-contained: main, includes, data structures, parsing helpers, and all logic in one source file. Do not emit a test harness or explanatory text. Return only the compilable task_queue.cpp source now.<|assistant|><think></think>
