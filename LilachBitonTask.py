from PosestTask import PosestTask
import matplotlib.pyplot as plt
import numpy as np

def InitialProcessing(target_class):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    for i in range(target_class.nframes-1):
        Frame1 = i
        Frame2 = Frame1 + 1
        pc1, _ = target_class.GetPts(Frame1)
        pc2, Pw2 = target_class.GetPts(Frame2)

        translation ,dth_real, dth_GT, normal_GT, P2_center_GT = target_class.RealMotion(Frame1,Frame2)   

        centroid_c1 = np.mean(pc1, axis=0)
        centroid_c2 = np.mean(pc2, axis=0)
        ax3.plot([centroid_c1[0],centroid_c1[1]],[centroid_c2[0],centroid_c2[1]]) 
        X, Y, Z, U, V, W = P2_center_GT[0],P2_center_GT[1],P2_center_GT[2],normal_GT[0],normal_GT[1], normal_GT[2]
        ax.scatter(Pw2.T[0],Pw2.T[2],Pw2.T[1])
        ax.quiver(X, Z, Y, U, W, V)      
        ax1.scatter(pc1.T[0],pc1.T[1])
        ax2.scatter(centroid_c1.T[0],centroid_c1.T[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Real World points and normal to each plane from points_world.pkl')
    ax.grid()

    ax1.set_xlabel('U')
    ax1.set_ylabel('V')
    ax1.set_title('Image points through frames from points_image.pkl')
    ax1.grid()

    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.set_title('C.G. Image points through frames from points_image.pkl')
    ax2.grid()

    plt.show() 


def Section1(img_pnts1, img_pnts2, Frame1, target):
    _, Pw_1 = target.GetPts(Frame1)
    # Estimation
    t_E, dth_E , pose_E, normal2_E, pnts_E, v_dir= target.MotionEstTwoFrame(img_pnts1,img_pnts2,Frame1,Pw_1)
    print('Estimation Results:')
    print('dx = ',t_E[0], 'dy = ',t_E[1], 'dz = ', t_E[2], 'dth = ', dth_E)
    # Ground Truth
    t_GT ,dth_real, dth_GT, normal_GT, P2_center_GT = target.RealMotion(Frame1,Frame1+1)
    print('Ground Truth:')
    print('dx = ',t_GT[0], 'dy = ',t_GT[1], 'dz = ', t_GT[2], 'dth = ', dth_GT,'[rad]')
    # Error calculation
    print('dxdydz Error = ',t_GT-t_E, 'dth Error', dth_GT-dth_E)

    # for plot
    centroid_Pw_1 = np.mean(Pw_1, axis=0)
    t_size = np.linalg.norm(t_E)
    X_e, Y_e, Z_e, U_e, V_e, W_e = pose_E.T[0],pose_E.T[1],pose_E.T[2],t_size*v_dir[0],t_size*v_dir[1], t_size*v_dir[2]
    
    move_line = np.array([centroid_Pw_1,centroid_Pw_1+t_E])
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X_e, Z_e, Y_e, U_e, W_e, V_e, color='r')
    ax.scatter(centroid_Pw_1.T[0],centroid_Pw_1.T[2],centroid_Pw_1.T[1])
    ax.scatter(pose_E.T[0],pose_E.T[2],pose_E.T[1])
    ax.plot(move_line.T[0],move_line.T[2],move_line.T[1],color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Real World points prediction and heading dirction for frames '+str(Frame1)+', '+str(Frame1+1))
    ax.grid()
    plt.show()

    return t_E , dth_E , pose_E, normal2_E, pnts_E

def Section2(target):
    e_t = []
    e_t_norm = []
    e_th = []
    _, Pw_1 = target.GetPts(0)
    p1_canter = np.mean(Pw_1, axis=0)
    # Estimation
    t_vec, dth_vec, center_vec, N_vec, dir_vec= target.MotionEstFull()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # Ground Truth
    for i in range(target.nframes-1):
        print('Estimation Results:')
        print('dx = ',t_vec[i,0], 'dy = ',t_vec[i,1], 'dz = ', t_vec[i,2], 'dth = ', dth_vec[i])
        
        t_GT ,dth_real, dth_GT, normal_GT, P2_center_GT = target.RealMotion(i,i+1)
        print('Ground Truth:')
        print('dx = ',t_GT[0], 'dy = ',t_GT[1], 'dz = ', t_GT[2], 'dth = ', dth_real,'[rad]')
        
        # Error calculation
        e_trans = t_GT - t_vec[i,:]
        e_dth = dth_real - dth_vec[i]
        print('dxdydz Error = ',e_trans, 'dth Error = ', e_dth)
        e_t.append(e_trans)
        e_th.append(abs(e_dth))
        e_t_norm.append(np.linalg.norm(abs(e_trans)))

        # for plot
        t_size = np.linalg.norm(t_vec[i,:])
        move_line = np.array([p1_canter,p1_canter+t_vec[i,:]])
        X_v, Y_v, Z_v, U_v, V_v, W_v = center_vec[i,0],center_vec[i,1],center_vec[i,2],dir_vec[i,0],dir_vec[i,1], dir_vec[i,2]
        
        ax.quiver(X_v, Z_v,Y_v, 0.2*t_size*U_v, 0.2*t_size*W_v, 0.2*t_size*V_v,color='r')
        ax.plot(move_line.T[0],move_line.T[2],move_line.T[1],'g')
        ax1.plot(move_line.T[0],move_line.T[2],'g')
        ax1.quiver(X_v, Z_v, 0.1*t_size*U_v, 0.1*t_size*W_v,color='r',width=0.005)
        p1_canter = P2_center_GT
    
    print('Max dth Error = ', np.max(e_th), 'Max translation Error = ', np.max(e_t_norm))

    ax1.scatter(center_vec.T[0],center_vec.T[2],s=10)
    ax1.grid()
    ax1.set_ylim([-1, 10])   
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Real World 2D points prediction and heading direction')
    ax.grid()
    ax.scatter(center_vec.T[0],center_vec.T[2],center_vec.T[1],s=5)
    ax.set_xlim([-11, -5])
    ax.set_ylim([-2, 11])
    ax.set_zlim([0.6, 0.8])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Real World 3D points prediction and heading direction')
    ax.grid()
    plt.show()

    return t_vec, dth_vec, center_vec, N_vec


def main():
    only_target_scenario = "only_target"
    target_ego_scenario = "target_and_ego"
    # Choose scenario!
    current_scenario = only_target_scenario
    # Creating class to this scenario
    target_class = PosestTask(current_scenario)
    # Initial Processing
    InitialProcessing(target_class)

    ## Section 1
    # Choose Frame!
    Frame1 = 5
    Frame2 = Frame1 + 1
    pc1, _ = target_class.GetPts(Frame1)
    pc2, _ = target_class.GetPts(Frame2)
    trans_W , dth , P2_center_E, normal2_E, P2_E_pnts= Section1(pc1 ,pc2 , Frame1, target_class)
    ## Section1 Maximal error Check
    e_t = []
    e_th = []
    for i in range(target_class.nframes-1):
        pc1, _ = target_class.GetPts(i)
        pc2, _ = target_class.GetPts(i+1)
        trans_W , dth , P2_center_E, normal2_E, P2_E_pnts= Section1(pc1 ,pc2 , i, target_class)
        t_GT ,dth_real, dth_GT, normal_GT, P2_center_GT = target_class.RealMotion(i,i+1)
        e_trans = abs(t_GT - trans_W)
        e_t.append(np.linalg.norm(e_trans))
        e_th.append(abs(dth_real-dth))
    print('Section1 Maximal Error:')
    print('Max dth Error = ', np.max(e_th), 'Max translation norm Error = ', np.max(e_t))

    ## Section 2
    t_vec, dth_vec, center_vec, N_vec = Section2(target_class)


if __name__ == "__main__":
    main()
