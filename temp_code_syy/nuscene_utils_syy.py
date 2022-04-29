            print('________________')
            # 逆向将yaw转为global坐标系下的rotation

            orientation = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=yaw)
            print(orientation)


            quaternion = Quaternion(calib_data['rotation'])
            center = np.dot(quaternion.rotation_matrix, center)
            center += np.array(calib_data['translation'])
            orientation = quaternion * orientation
            print("back_ego_center:", center)
            print("back_ego_orientation:", orientation)
            print("--")


            quaternion = Quaternion(ego_data['rotation'])
            center = np.dot(quaternion.rotation_matrix, center)
            center += np.array(ego_data['translation'])
            orientation = quaternion * orientation
            orientation = np.around(np.array([i for i in orientation]), 7)
            print("back_center:", center)
            print("back_orientation:", orientation)
            print("--")
            break