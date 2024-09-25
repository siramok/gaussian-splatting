from plyfile import PlyData

# Path to your binary PLY file
binary_ply_file_path = "point_clouds/point_cloud.ply"

# Load the binary PLY file
ply_data = PlyData.read(binary_ply_file_path)

# Path for the ASCII PLY file output
ascii_ply_file_path = "point_clouds/point_cloud_ascii.ply"

# Open the output file in write mode
with open(ascii_ply_file_path, "w") as f:
    # Write the PLY header in ASCII format
    f.write(f"ply\n")
    f.write(f"format ascii 1.0\n")

    # Write all element declarations (from the header)
    for element in ply_data.elements:
        f.write(f"element {element.name} {element.count}\n")
        for prop in element.properties:
            f.write(f"property float {prop.name}\n")

    # Write the end of the header
    f.write(f"end_header\n")

    # Write all the data for each element in ASCII format
    for element in ply_data.elements:
        for row in element.data:
            # Convert each row of binary data to ASCII and write to file
            f.write(" ".join(str(val) for val in row) + "\n")

print(f"Converted binary PLY file to ASCII format: {ascii_ply_file_path}")
