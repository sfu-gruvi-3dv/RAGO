import torch
import torch.nn.functional as F
import numpy as np

from model.utils import edge_model_rot
from model import tools as tools
from scipy.sparse import csr_matrix
from torch_scatter import scatter
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
def rel_rot_from_global(node, edge_index):
    row, col = edge_index
    ri, rj = node[row], node[col]
    rij = torch.bmm(ri, rj.transpose(1,2))
    return rij

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def inv_q(q):
    """
    Inverse quaternion(s) q .
    """
    assert q.shape[-1] == 4
    original_shape = q.shape
    q = q.view(-1,4)
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=1).view(original_shape)


def rot2quaternion(rot_mat, eps=1e-6):
    """ Convert 3x4 rotation matrix to 4d quaternion vector [From torchgeometrc library]

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rot_mat (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rot_mat_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rot_mat):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rot_mat)))

    if len(rot_mat.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rot_mat.shape))

    rot_mat = rot_mat[:, :3, :3]
    rmat_t = torch.transpose(rot_mat, 1, 2)

    mask_d2 = (rmat_t[:, 2, 2] < eps).float()
    mask_d0_d1 = (rmat_t[:, 0, 0] > rmat_t[:, 1, 1]).float()
    mask_d0_nd1 = (rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]).float()

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_rel_deg(q1, q2):
    torchpi = torch.acos(torch.zeros(1)).item() *2
    delta_q = qmul(inv_q(q1), q2)
    delta_q = F.normalize(delta_q, p=2, dim=1)
    theta = 2.0 * torch.arccos(torch.abs(delta_q[:,0])) * 180.0 / torchpi 
    return theta.view(q1.shape[0],1)

def quaternion2deg(q1):
    torchpi = torch.acos(torch.zeros(1)).item() *2
    delta_q = F.normalize(q1, p=2, dim=1)
    theta = 2.0 * torch.arccos(torch.abs(delta_q[:,0])) * 180.0 / torchpi 
    return theta.view(q1.shape[0],1)

def quaternion2rot(q):
    """
    :param q: normalized quaternion vector, dim: (N, 4)
    :return: rotation matrix, dim: (N, 3, 3)
    """
    N = q.shape[0]
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    return torch.stack([1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw,
                        2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw,
                        2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy
                        ], dim=1).view(N, 3, 3)
    
    
#T_poses num*3
#r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch=r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch,1, 3,3).expand(batch,joint_num, 3,3).contiguous().view(batch*joint_num,3,3)
    src_poses = T_pose.view(1,joint_num,3,1).expand(batch,joint_num,3,1).contiguous().view(batch*joint_num,3,1)
        
    out_poses = torch.matmul(r_matrices, src_poses) #(batch*joint_num)*3*1
        
    return out_poses.view(batch, joint_num,3)
    
# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1) + 1e-8)# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#in batch*6
#out batch*5
def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:,0:dim-1]/(1-a[:,dim-1])
    return out
	


#in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans


#a batch*5
#out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2)+1, np.sqrt(2)+1, np.sqrt(2)]) #3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1,3).repeat(batch,1) #batch,3
    
    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)#batch*4
    norm = torch.sqrt(torch.pow(u[:,1:],2).sum(1)) #batch
    u = u/ norm.view(batch,1).repeat(1,u.shape[1]) #batch*4
    b = torch.cat((a[:,0:2], u),1)#batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


#quaternion batch*4
def compute_rotation_matrix_from_quaternion( quaternion):
    batch=quaternion.shape[0]
    
    
    quat = normalize_vector(quaternion).contiguous()
    
    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    
#axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle( axisAngle):
    batch = axisAngle.shape[0]
    
    theta = torch.tanh(axisAngle[:,0])*np.pi #[-180, 180]
    sin = torch.sin(theta*0.5)
    axis = normalize_vector(axisAngle[:,1:4]) #batch*3
    qw = torch.cos(theta*0.5)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez( rod):
    batch = rod.shape[0]
    
    axis, theta = normalize_vector(rod, return_mag=True)
    
    sin = torch.sin(theta)
    
    
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf( hopf):
    batch = hopf.shape[0]
    
    theta = (torch.tanh(hopf[:,0])+1.0)*np.pi/2.0 #[0, pi]
    phi   = (torch.tanh(hopf[:,1])+1.0)*np.pi     #[0,2pi)
    tao   = (torch.tanh(hopf[:,2])+1.0)*np.pi     #[0,2pi)
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#euler_sin_cos batch*6
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch=euler_sin_cos.shape[0]
    
    s1 = euler_sin_cos[:,0].view(batch,1)
    c1 = euler_sin_cos[:,1].view(batch,1)
    s2 = euler_sin_cos[:,2].view(batch,1)
    c2 = euler_sin_cos[:,3].view(batch,1)
    s3 = euler_sin_cos[:,4].view(batch,1)
    c3 = euler_sin_cos[:,5].view(batch,1)

        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    return theta


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    
    batch=m.shape[0]
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    theta = torch.acos(cos)
    
    return theta
    
def get_sampled_rotation_matrices_by_quat(batch):
    #quat = torch.autograd.Variable(torch.rand(batch,4).cuda())
    quat = torch.autograd.Variable(torch.randn(batch, 4).cuda())
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix
    
def get_sampled_rotation_matrices_by_hpof(batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,1, batch)*np.pi).cuda()) #[0, pi]
    phi   =  torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    tao   = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle( batch, return_quaternion=False):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(-1,1, batch)*np.pi).cuda()) #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    quaternion = torch.cat((qw.view(batch,1), qx.view(batch,1), qy.view(batch,1), qz.view(batch,1)), 1 )
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    if(return_quaternion==True):
        return matrix, quaternion
    else:
        return matrix

def random_sample_subgraph_from_data_batch_rotation(data_batch, ratio=0.8, is_connected=False, connected_mode="gt"):
    select = torch.zeros(data_batch.edge_index.shape[1]).to(data_batch.edge_index.device)
    if is_connected:
        # using connected mode to choose which method to use.
        # 1. gt will use gt_rot to make sure the edges in the best spt will be selected in training set
        # 2. est will use ? to make sure the graph is connected.

        node_quat =data_batch.gt_rot
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr

        is_quat = False
        if node_quat.shape[1] == 4:
            is_quat = True
            node_quat = quaternion2rot(node_quat)
            edge_attr = quaternion2rot(edge_attr)

        rel_qaut = edge_model_rot(node_quat, edge_index)
        rel_deg = tools.compute_geodesic_distance_from_two_matrices(rel_qaut.view(-1,3,3), edge_attr.view(-1,3,3)).view(-1)
        adj_deg = torch.zeros(node_quat.shape[0], node_quat.shape[0]).to(node_quat.device)
        adj_ind = torch.zeros(node_quat.shape[0], node_quat.shape[0]).to(node_quat.device)
        adj_deg[edge_index[0], edge_index[1]] = rel_deg.view(-1)
        adj_ind[edge_index[0],edge_index[1]] = torch.arange(0, edge_index.shape[1], step=1).float().to(node_quat.device)

        X = csr_matrix(adj_deg.detach().cpu().numpy().tolist())
        mst = minimum_spanning_tree(X)
        mst = mst.toarray().astype(float)
        mst = torch.tensor(mst) != 0.
        mst_idx = adj_ind[(mst | mst.T)]
        edge_ind = mst_idx.long()
        select[edge_ind] = 1
    select = select.bool()
    thres = 1.0 - ratio
    edge_prob = torch.rand(data_batch.edge_index.shape[1]//2) > thres
    edge_prob = edge_prob.to(select.device)
    ind = torch.arange(0, data_batch.edge_index.shape[1], 2).long()
    select[ind] = edge_prob
    select[ind+1] = edge_prob
    if is_connected:
        select[edge_ind] = 1



    data_batch.now_edge = select
    data_batch.edge_attr = data_batch.edge_attr[select,:]
    data_batch.edge_index = data_batch.edge_index[:, select]
    data_batch.valid_edge = data_batch.valid_edge.view(-1)[select]
    return data_batch

    

    