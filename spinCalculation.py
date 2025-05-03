#!/usr/bin/env python3
"""
spinCalculation.py – Estimate golf-ball spin from two grayscale frames
---------------------------------------------------------------------
* **Automatic mode** (default) – ORB key-points + brute-force matching
* **Manual mode**      `--manual`    – Interactive clicking of corresponding dimples
* **Headless manual**  `--points`    – Text file `x1 y1 x2 y2` (one pair/line)

Outputs (in `DEBUG_DIR`):
* `predicted_next_frame.png` – Synthetic next-frame under same rotation
* `manual_matches.png`      – Colour-coded match graph (manual mode)

Console prints:
* Pixel coords, unit-sphere vectors, rotation matrix, axis-angle, angular speed (rad/s and RPM)
* Final spin breakdown: back-spin, side-spin, spin-axis elevation

© ChatGPT (o3) 2025-05-02 – MIT-ish licence, use freely.
"""
from __future__ import annotations
import cv2, numpy as np, os, math, argparse, sys
from dataclasses import dataclass
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Default paths – override via CLI
IMG1      = "/home/vglalala/GCFive/Images/spin_ball_1_gray_image1.png"
IMG2      = "/home/vglalala/GCFive/Images/spin_ball_2_gray_image1.png"
DEBUG_DIR = "/home/vglalala/GCFive/debug"
DELTA_T   = 1/600.0                      # seconds between frames (e.g. 600 FPS)

# ---------------------------------------------------------------------------
@dataclass
class SpinResult:
    backspin_rpm:  float
    sidespin_rpm:  float
    axis_elev_deg: float
    rot_mat:       np.ndarray
    angle_rad:     float

# ---------------- Pretty print helpers -------------------------------------

def pvec(name: str, v: np.ndarray):
    print(f"{name}:", np.array2string(v, formatter={'float': '{: 10.6f}'.format}))

def header(txt: str):
    print("\n" + "="*len(txt))
    print(txt)
    print("="*len(txt))

# ---------------- Ball localisation (crop & mask) --------------------------

def crop_ball(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int]]:
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=100,
                               param1=120, param2=30,
                               minRadius=40, maxRadius=0)
    if circles is None:
        raise RuntimeError("Could not find ball – adjust Hough parameters.")
    x, y, r = map(int, circles[0,0])
    roi = gray[y-r:y+r, x-r:x+r].copy()
    mask = np.zeros_like(roi)
    cv2.circle(mask, (r,r), r, 255, -1)
    roi = cv2.bitwise_and(roi, roi, mask=mask)
    return roi, (x, y, r)

# ---------------- ORB keypoints & matching ---------------------------------

def orb_kp_desc(img: np.ndarray):
    orb = cv2.ORB_create(2000)
    return orb.detectAndCompute(img, None)

def brute_force_match(d1, d2, top_k: int = 200):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
    return matches[:top_k]

# ---------------- Geometry & rotation math ---------------------------------

def kps_to_sphere(pts: List[Tuple[float,float]], r: int) -> np.ndarray:
    out = []
    for x,y in pts:
        xn, yn = x/r, y/r
        if xn*xn + yn*yn >= 1: continue
        zn = math.sqrt(1 - xn*xn - yn*yn)
        out.append((xn, yn, zn))
    return np.asarray(out, dtype=np.float32)

def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    H = A.T @ B
    U,_,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2] *= -1
        R = Vt.T @ U.T
    return R

def rotation_to_spin(R: np.ndarray, dt: float) -> SpinResult:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R)-1)/2)))
    if angle < 1e-7:
        return SpinResult(0,0,0,R,0)
    axis = np.array([(R[2,1]-R[1,2]), (R[0,2]-R[2,0]), (R[1,0]-R[0,1])])/(2*math.sin(angle))
    omega = axis*angle/dt
    rpm   = omega*60/(2*math.pi)
    elev  = math.degrees(math.acos(abs(axis[2])))
    return SpinResult(rpm[0], rpm[1], elev, R, angle)

def rotate_roi_once(roi: np.ndarray, R: np.ndarray) -> np.ndarray:
    h,w = roi.shape
    cx = cy = r = w//2
    yy,xx = np.indices((h,w))
    xn = (xx-cx)/r; yn = (yy-cy)/r
    mask = xn*xn+yn*yn <= 1
    zn = np.sqrt(np.clip(1-(xn*xn+yn*yn),0,1))
    xyz = np.dstack((xn,yn,zn))
    xyzr = xyz @ R.T
    x2 = (xyzr[...,0]*r+cx).astype(np.float32)
    y2 = (xyzr[...,1]*r+cy).astype(np.float32)
    map_x = np.where(mask,x2,-1).astype(np.float32)
    map_y = np.where(mask,y2,-1).astype(np.float32)
    return cv2.remap(roi,map_x,map_y,interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)

# ---------------- Manual point-pair GUI ------------------------------------

def collect_pairs(roi1: np.ndarray, roi2: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    pad = 20
    vis = np.hstack((roi1, np.full((roi1.shape[0],pad),200,np.uint8), roi2))
    h,w = vis.shape; w1 = roi1.shape[1]
    pts1, pts2 = [],[]; idx=0
    win = "Click pairs (left→right) | u:undo  Enter:done  Esc:quit"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    def cb(ev,x,y,fl,ud):
        nonlocal idx
        if ev==cv2.EVENT_LBUTTONDOWN:
            if x<w1:
                pts1.append((x,y)); cv2.circle(vis,(x,y),4,255,-1);
                cv2.putText(vis,str(idx+1),(x+6,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,255,1)
            elif x>w1+pad:
                xr=x-(w1+pad); pts2.append((xr,y));
                cv2.circle(vis,(x,y),4,0,-1);
                cv2.putText(vis,str(idx+1),(x+6,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,0,1);
                idx+=1
            cv2.imshow(win,vis)
    cv2.setMouseCallback(win,cb); cv2.imshow(win,vis)
    while True:
        k=cv2.waitKey(0)&0xFF
        if k in (13,10): break
        if k==27: sys.exit("Aborted.")
        if k in (ord('u'),ord('U')) and idx>0:
            idx-=1; pts1.pop(); pts2.pop(); vis[:]=0
            vis[:]=np.hstack((roi1,np.full((h,pad),200,np.uint8),roi2))
            for i,(p1,p2) in enumerate(zip(pts1,pts2),1):
                cv2.circle(vis,p1,4,255,-1); cv2.putText(vis,str(i),(p1[0]+6,p1[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,255,1)
                dp=(p2[0]+w1+pad,p2[1]); cv2.circle(vis,dp,4,0,-1)
                cv2.putText(vis,str(i),(dp[0]+6,dp[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,0,1)
            cv2.imshow(win,vis)
    cv2.destroyWindow(win)
    if len(pts1)<4 or len(pts1)!=len(pts2): sys.exit("Need ≥4 pairs.")
    return np.array(pts1,np.float32),np.array(pts2,np.float32)

# ---------------- Solver implementations ----------------------------------

def solve_spin_auto(roi1,roi2,r1,r2,dt):
    kp1,d1=orb_kp_desc(roi1); kp2,d2=orb_kp_desc(roi2)
    matches=brute_force_match(d1,d2)
    Apx=[(kp1[m.queryIdx].pt[0]-r1,kp1[m.queryIdx].pt[1]-r1) for m in matches]
    Bpx=[(kp2[m.trainIdx].pt[0]-r2,kp2[m.trainIdx].pt[1]-r2) for m in matches]
    A=kps_to_sphere(Apx,r1); B=kps_to_sphere(Bpx,r2)
    R=kabsch(A,B)
    return rotation_to_spin(R,dt)


def solve_spin_manual(roi1,roi2,r,dt):
    pts1,pts2=collect_pairs(roi1,roi2)
    header("Manual point pairs (pixel)")
    for i,(p1,p2) in enumerate(zip(pts1,pts2),1):
        print(f"{i:2d}: ({p1[0]:5.1f},{p1[1]:5.1f}) ↔ ({p2[0]:5.1f},{p2[1]:5.1f})")
    A=kps_to_sphere([(x-r,y-r) for x,y in pts1],r)
    B=kps_to_sphere([(x-r,y-r) for x,y in pts2],r)
    header("Unit-sphere vectors")
    for i,(a,b) in enumerate(zip(A,B),1): pvec(f"A{i}",a); pvec(f"B{i}",b)
    R=kabsch(A,B)
    header("Rotation matrix R")
    print(np.array2string(R,formatter={'float':'{:10.6f}'.format}))
    spin=rotation_to_spin(R,dt)
    header("Axis-angle & angular velocity")
    axis=np.array([(R[2,1]-R[1,2]),(R[0,2]-R[2,0]),(R[1,0]-R[0,1])])/(2*math.sin(spin.angle_rad))
    pvec("Axis (unit)",axis)
    print(f"Angle Δθ        : {math.degrees(spin.angle_rad):.4f}°")
    print(f"ω (rad/s)       : {spin.angle_rad/dt:.2f}")
    print(f"ω (rpm)         : {spin.angle_rad*60/(2*math.pi*dt):.2f}")
    # save matches image
    pad=20; w1=roi1.shape[1]
    vis=np.hstack((roi1,np.full((roi1.shape[0],pad),220,np.uint8),roi2))
    cols=[(255,0,0),(0,0,255),(0,255,0),(0,255,255),(255,0,255),(255,255,0)]
    for i,(p1,p2) in enumerate(zip(pts1,pts2)):
        c=cols[i%len(cols)]
        cv2.circle(vis,(int(p1[0]),int(p1[1])),4,c,-1)
        dp=(int(p2[0]+w1+pad),int(p2[1]))
        cv2.circle(vis,dp,4,c,-1)
        cv2.line(vis,(int(p1[0]),int(p1[1])),dp,c,1)
        cv2.putText(vis,str(i+1),(int(p1[0])+6,int(p1[1])-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,c,1)
        cv2.putText(vis,str(i+1),(dp[0]+6,dp[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,c,1)
    os.makedirs(DEBUG_DIR,exist_ok=True)
    path=os.path.join(DEBUG_DIR,"manual_matches.png")
    cv2.imwrite(path,vis)
    print(f"Matches graph → {path}")
    return spin


def solve_spin_from_arrays(pts1,pts2,r,dt):
    header("Manual point pairs (pixel) [from file]")
    for i,(p1,p2) in enumerate(zip(pts1,pts2),1):
        print(f"{i:2d}: ({p1[0]:5.1f},{p1[1]:5.1f}) ↔ ({p2[0]:5.1f},{p2[1]:5.1f})")
    A=kps_to_sphere([(x-r,y-r) for x,y in pts1],r)
    B=kps_to_sphere([(x-r,y-r) for x,y in pts2],r)
    header("Unit-sphere vectors")
    for i,(a,b) in enumerate(zip(A,B),1): pvec(f"A{i}",a); pvec(f"B{i}",b)
    R=kabsch(A,B)
    header("Rotation matrix R")
    print(np.array2string(R,formatter={'float':'{:10.6f}'.format}))
    spin=rotation_to_spin(R,dt)
    return spin

# ---------------- Main entry ------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img1",default=IMG1)
    ap.add_argument("--img2",default=IMG2)
    ap.add_argument("--delta_t",type=float,default=DELTA_T,help="Frame interval (s)")
    ap.add_argument("--manual","-m",action="store_true",help="Interactive manual mode")
    ap.add_argument("--points",help="Text file with x1 y1 x2 y2 pairs")
    args=ap.parse_args()

    os.makedirs(DEBUG_DIR,exist_ok=True)
    g1=cv2.imread(args.img1,cv2.IMREAD_GRAYSCALE)
    g2=cv2.imread(args.img2,cv2.IMREAD_GRAYSCALE)
    roi1,(cx1,cy1,r1)=crop_ball(g1)
    roi2,(cx2,cy2,r2)=crop_ball(g2)

    # Choose solver
    if args.points:
        pts=np.loadtxt(args.points,dtype=np.float32)
        if pts.ndim!=2 or pts.shape[1]!=4 or pts.shape[0]<4:
            sys.exit("--points needs >=4 rows of x1 y1 x2 y2")
        pts1,pts2=pts[:,:2],pts[:,2:]
        spin=solve_spin_from_arrays(pts1,pts2,r1,args.delta_t)
    elif args.manual:
        spin=solve_spin_manual(roi1,roi2,r1,args.delta_t)
    else:
        spin=solve_spin_auto(roi1,roi2,r1,r2,args.delta_t)

    # Summary
    sign=lambda v:'+' if v>=0 else '-'
    print(f"\nΔt     = {args.delta_t*1e3:.2f} ms")
    print(f"Back-spin         : {sign(spin.backspin_rpm)}{abs(spin.backspin_rpm):.0f} rpm")
    print(f"Side-spin         : {sign(spin.sidespin_rpm)}{abs(spin.sidespin_rpm):.0f} rpm")
    print(f"Spin-axis elev    : {spin.axis_elev_deg:.1f} °")

    # Predict next frame
    pred=rotate_roi_once(roi1,spin.rot_mat)
    canvas=np.zeros_like(g1)
    canvas[cy1-r1:cy1+r1,cx1-r1:cx1+r1]=pred
    outp=os.path.join(DEBUG_DIR,"predicted_next_frame.png")
    cv2.imwrite(outp,canvas)
    print(f"Predicted frame → {outp}\n")

if __name__=='__main__':
    main()