import pandas as pd
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

class PosestTask:
    def __init__(self, scenario):
        ## Loading the data
        self.directory = './' + scenario + '/'
        self.world_df = pd.read_pickle(self.directory + 'points_world.pkl')
        self.img_df = pd.read_pickle(self.directory + 'points_image.pkl')
        self.mat_df = pd.read_pickle(self.directory + 'conversion_matrices.pkl')
        with open(self.directory + "conf.yaml", 'r') as stream:
            try:
                self.conf = yaml.safe_load(stream)
                self.K = np.array(self.conf["K"])
                self.nframes = self.conf["nframes"]
                self.npnts = self.conf["npnts"]
            except yaml.YAMLError as exc:
                print(exc)

    def GetMat(self, Frame):
        frame_mat = self.mat_df.query('Frame==' + str(Frame))
        T_c_e = frame_mat['T_c_e'][Frame]
        T_e_e0 = frame_mat['T_e_e0'][Frame]
        T_t_t0 = frame_mat['T_t_t0'][Frame]
        return T_c_e, T_e_e0, T_t_t0

    def GetPts(self, Frame):
        frame_img = self.img_df.query('Frame==' + str(Frame))
        img_p=np.vstack((frame_img['U'].values,frame_img['V'].values)).T
        frame_world = self.world_df.query('Frame==' + str(Frame))
        world_P=np.vstack((frame_world['X'].values,frame_world['Y'].values,frame_world['Z'].values)).T
        return img_p , world_P
    
    def FindNormal(self, P1, P2, P3):
        pq = P2 - P1 
        pr = P3 - P1
        cross_pro = np.cross(pq,pr)
        n = cross_pro/np.linalg.norm(cross_pro)

        return n

    def RealMotion(self, Frame1 , Frame2):
        _, _, T_t1_t0 = self.GetMat(Frame1)
        _, _, T_t2_t0 = self.GetMat(Frame2)
        _, Pw_1 = self.GetPts(Frame1)
        _, Pw_2 = self.GetPts(Frame2)
        normal1 = self.FindNormal(Pw_1[0,:] ,Pw_1[1,:], Pw_1[2,:])
        normal2 = self.FindNormal(Pw_2[0,:] ,Pw_2[1,:], Pw_2[2,:])
        dot_prod = np.dot(normal2,normal1)
        if abs(dot_prod-1) < 0.000001:
            dot_prod = 1
        dth_true = -np.arccos(dot_prod)
        centroid_w1 = np.mean(Pw_1, axis=0)
        centroid_w2 = np.mean(Pw_2, axis=0)
        translation = centroid_w2 - centroid_w1
        R_t1_t2 = T_t1_t0[:3,:3] @ T_t2_t0[:3,:3].T
        euler =  (R.from_matrix(R_t1_t2)).as_rotvec()    
        return translation , euler[1] , dth_true ,normal2 , centroid_w2
  
    def MotionEstTwoFrame(self,pc0,pc1, Frame1, P0):
        Pw1_e0_Est = []
        z_c = []
        x_c = []
        y_c = []
        f = self.K[0,0]
        centroid_w0 = np.mean(P0, axis=0)
        T_c1_e, T_e1_e0, _ = self.GetMat(Frame1)
        T_c2_e, T_e2_e0, _ = self.GetMat(Frame1+1)
        # transform all points from image to ego_0
        for i in range(self.npnts):
            P_w0_c = (((T_c1_e @ T_e1_e0)) @ np.append(P0[i,:], 1))[:3]
            z_c.append(f*P_w0_c[1]/pc1[i,1])
            x_c.append((pc1[i,0]*z_c[i])/f)
            y_c.append(P_w0_c[1])
            Pw1_c_4 = np.array([x_c[i],y_c[i],z_c[i],1])
            Pw1_e0_Est.append(((np.linalg.pinv(T_e2_e0) @ np.linalg.pinv(T_c2_e)) @ Pw1_c_4)[:3])
        # calc translation of C.G. points in real world
        Pw1_e0_Est = np.asarray(Pw1_e0_Est)
        Pw1_Est = np.mean(Pw1_e0_Est, axis=0)
        trans_w = Pw1_Est - centroid_w0 
        # calc orientation using angle between normals
        normal1 = self.FindNormal(P0[0,:] ,P0[1,:], P0[2,:])
        normal2 = self.FindNormal(Pw1_e0_Est[0,:] ,Pw1_e0_Est[1,:], Pw1_e0_Est[2,:])
        dot_prod = np.dot(normal2,normal1)
        if abs(dot_prod-1) < 0.000001:
            dot_prod = 1
        dth = -np.arccos(dot_prod)
        # calc moving direction (perpendicular to normal)       
        p2 = [(1+(normal2[2]/normal2[0])**2), (2*normal2[2]*normal2[1]**2)/normal2[0]**2,(normal2[1]**4/normal2[0]**2)-1]
        roots2 = np.roots(p2)
        v_3_2 = np.max(roots2)
        v_1_2 = -v_3_2*(normal2[2]/normal2[0])-(normal2[1]**2/normal2[0])
        v_dir = np.array([v_1_2, normal2[1], v_3_2])

        return trans_w, dth , Pw1_Est , normal2 , Pw1_e0_Est, v_dir

    def MotionEstFull(self):
        f = self.K[0,0]
        t_vec = []
        dth_vec = []
        dir_vec = []
        center_vec =[]
        N_vec = []
        P2_E_pnts = []
        for i in range(self.nframes-1):
            frame1 = i
            if (frame1%5) == 0:
                pc0 , Pw0 = self.GetPts(frame1)
                pc1 , _ = self.GetPts(frame1+1)
                if frame1 != 0:
                    error = P2_E_pnts - Pw0
                    print('Norm of Predicrion error:',np.linalg.norm(abs(error)))
                trans_W , dth , P2_center_E, normal2_E, P2_E_pnts, v_dir= self.MotionEstTwoFrame(pc0, pc1, frame1,Pw0)
            else:
                P1_e = P2_E_pnts.copy()
                pc0 , _ = self.GetPts(frame1)
                pc1 , _ = self.GetPts(frame1+1)
                trans_W , dth , P2_center_E, normal2_E, P2_E_pnts, v_dir= self.MotionEstTwoFrame(pc0, pc1, frame1,P1_e)

            t_vec.append(trans_W)
            dth_vec.append(dth)
            center_vec.append(P2_center_E)
            N_vec.append(normal2_E)
            dir_vec.append(v_dir)
            
        t_vec = np.asarray(t_vec)
        dth_vec = np.asarray(dth_vec)
        center_vec = np.asarray(center_vec)
        N_vec = np.asarray(N_vec)
        dir_vec = np.asarray(dir_vec)

        return t_vec, dth_vec, center_vec, N_vec, dir_vec

