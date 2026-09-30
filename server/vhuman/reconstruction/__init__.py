"""Original portrait fitting, explicit materials and mesh-bound appearance.

Heavy dependencies are loaded only by the offline worker. Candidate outputs
never replace the accepted subject or its trained deformation packages.
"""

FILES = ('manifest.json','observations.json','geometry.npz','skin_material.json',
         'skin_basecolor.png','skin_orm.png','skin_normal.png','skin_specular.png',
         'skin_coverage.png','skin_confidence.png','gaussians.json','depth.npy','depth.json',
         'portrait.png','preview.png')
