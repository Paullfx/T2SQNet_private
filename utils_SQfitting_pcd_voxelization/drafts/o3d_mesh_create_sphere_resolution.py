import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=4)

o3d.visualization.draw_geometries([mesh])