from make_class import Camera_xyz

if __name__ == "__main__":
    a = Camera_xyz()

    name, x, y, z = a.object_detected('tv') # 찾을 물품
    
    print(name, x, y, z)

