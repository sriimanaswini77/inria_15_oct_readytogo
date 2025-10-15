import cv2
import numpy as np
from glob import glob
from math import atan, degrees
from datetime import datetime
from inria_qatm_pytorch_v2_copy_UPDATED import *
import argparse
from skimage.draw import line
from noise import noise, blur
from cam_sim import SimCamera
import torchvision
from time import time,sleep

FPS = 30
source_image_scale_factor = 1
select_points = True

def mouse_click(event, x, y, flags, param):
    global scaled_image, points_array, source_image_scale_factor, line_draw_array, select_points
    if select_points:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(scaled_image, (x, y), 5, (0, 0, 255), 4)
            points_array.append([x * source_image_scale_factor, y * source_image_scale_factor])
            line_draw_array.append([x, y])
            if len(points_array) > 1:
                delta_x, delta_y = line_draw_array[-2][0] - line_draw_array[-1][0], line_draw_array[-2][1] - line_draw_array[-1][1]
                ang = degrees(atan(delta_y / delta_x))
                print(line_draw_array[-2], line_draw_array[-1], ang)
                cv2.line(scaled_image, tuple(line_draw_array[-2]), tuple(line_draw_array[-1]), (0, 255, 255), 2)
            cv2.imshow("point_selector", scaled_image)

def rect_bbox2(rect):
    (center_x, center_y), (width, height), _ = rect
    x, y, w, h = int(center_x - width / 2), int(center_y - height / 2), int(width), int(height)
    return (x, y, x + w, y + h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='INRIA QATM with DYNAMIC INS Point and Search Region Visualization')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resize', '-r', type=int, default=100)
    parser.add_argument('--crop_size', '-cs', type=int, default=150)
    parser.add_argument('--alpha', '-a', type=float, default=25)
    parser.add_argument('--fps', '-f', type=int, default=5)
    parser.add_argument('--scale_factor', '-sf', type=int, default=1)
    parser.add_argument('--thres', '-t', type=float, default=0.7, help="threshold for QATM matching")
    parser.add_argument('--source', '-s', type=str,
                        default=r'C:\Users\manas\Downloads\kalidasu sir work\manaswini\satellite_1k.jpg')
    parser.add_argument('--noise', '-n', type=str, default='none',
                        help="Noise type {gauss_n, gauss_u, sp, poisson, random, none}")
    parser.add_argument('--blur', '-b', type=str, default='none',
                        help="blur type = {normal, median, gauss, bilateral, none}")
    parser.add_argument('--blur_filter', '-bf', type=int, default=5,
                        help="blur filter size, must be odd number")
    parser.add_argument('--local', '-l', action='store_true', default=False)
    parser.add_argument('--local_size', '-ls', type=int, default=300)

    # ARGUMENTS FOR DYNAMIC INS POINT
    parser.add_argument('--search_region_width', '-srw', type=int, default=500,
                        help="Width of search region around INS point")
    parser.add_argument('--search_region_height', '-srh', type=int, default=500,
                        help="Height of search region around INS point")
    parser.add_argument('--ins_offset_x', '-iox', type=int, default=0,
                        help="X offset of INS point from template point (initial only)")
    parser.add_argument('--ins_offset_y', '-ioy', type=int, default=0,
                        help="Y offset of INS point from template point (initial only)")
    parser.add_argument('--dynamic_update', '-du', action='store_true', default=True,
                        help="Enable dynamic INS point update based on detected matches")
    parser.add_argument('--show_search_region', '-ssr', action='store_true', default=True,
                        help="Show search region rectangle on output")

    args = parser.parse_args()
    print(args)

    width, height = args.crop_size, args.crop_size

    # random seed to reproduce the result
    np.random.seed(123)

    source_image_scale_factor = args.scale_factor
    points_array = []
    line_draw_array = []
    template_resolution = (width, height)

    # Load source image
    source_image = cv2.imread(args.source)
    src_h, src_w = source_image.shape[:2]
    scaled_image = cv2.resize(source_image, (src_w * source_image_scale_factor, src_h * source_image_scale_factor))

    # Point selection interface
    print("\n" + "="*70)
    print("POINT SELECTION MODE - DYNAMIC INS TRACKING WITH VISUALIZATION")
    print("="*70)
    print("Click on the image to select path points.")
    print("The FIRST point will be used as the INITIAL template point.")
    print("Cyan rectangle will show the active search region.")
    print("Press any key when done selecting points.")
    print("="*70 + "\n")

    cv2.imshow("point_selector", scaled_image)
    cv2.setMouseCallback('point_selector', mouse_click)
    cv2.waitKey()
    select_points = False
    cv2.destroyWindow("point_selector")

    print("\n" + "="*70)
    print("SELECTED POINTS:")
    print("="*70)
    print(f"Total points selected: {len(points_array)}")
    print("Points array:", points_array)
    print("="*70 + "\n")

    # ========== DEFINE INITIAL TEMPLATE POINT AND INS POINT ==========
    if len(points_array) == 0:
        print("ERROR: No points selected! Exiting...")
        exit()

    # Use the FIRST point as the initial template point
    template_point_index = 0
    template_point = points_array[template_point_index]

    # Define INITIAL INS point based on template point + offsets
    current_ins_point = (
        int(template_point[0] + args.ins_offset_x),
        int(template_point[1] + args.ins_offset_y)
    )

    print("="*70)
    print("INITIAL INS POINT CONFIGURATION:")
    print("="*70)
    print(f"Template Point (index {template_point_index}): {template_point}")
    print(f"Initial INS Point: {current_ins_point}")
    print(f"Search Region Size: {args.search_region_width}x{args.search_region_height}")
    print(f"Dynamic Update Mode: {'ENABLED ✓' if args.dynamic_update else 'DISABLED'}")
    print(f"Search Region Visualization: {'ENABLED ✓' if args.show_search_region else 'DISABLED'}")
    print("="*70 + "\n")

    # ========== INITIALIZE DATA LOADER WITH INITIAL INS POINT ==========
    print("Initializing ImageData with initial INS point...")
    data_loader = ImageData(
        source_img=args.source, 
        half=False, 
        thres=args.thres,
        ins_point=current_ins_point,
        search_region_size=(args.search_region_width, args.search_region_height)
    )
    print("ImageData initialized successfully!\n")

    # ========== LOAD MODEL ==========
    print("Loading VGG19 model...")
    model = torchvision.models.vgg19()
    try:
        model.load_state_dict(torch.load("./vgg19.pth"))
    except:
        print("Could not load vgg19.pth, using pretrained weights...")
        from torchvision.models import vgg19, VGG19_Weights
        model = vgg19(weights=VGG19_Weights.DEFAULT)

    model = model.eval()
    from torchvision.models import vgg19, VGG19_Weights
    model = CreateModel(25, vgg19(weights=VGG19_Weights.DEFAULT).features, use_cuda=torch.cuda.is_available())
    print("Model loaded successfully!\n")

    # Setup visualization
    color_list = color_palette("hls", 1)
    color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))
    color = 0
    d_img = data_loader.image_raw.copy()
    new_size = (int(width * args.resize / 100.0), int(height * args.resize / 100.0))

    # ========== START CAMERA SIMULATION ==========
    print("Starting camera simulation...")
    camera_fps = args.fps
    camera = SimCamera(points_array=points_array, image=args.source, fps=camera_fps, skip_pts=3)
    camera.start()
    print("Camera started!\n")

    print("="*70)
    print("DYNAMIC TEMPLATE MATCHING STARTED")
    print("="*70)
    print("Search region will PROPAGATE based on detected matches.")
    print("Cyan rectangle shows the active search region.")
    print("Press 'q' in the CV window to quit.\n")

    tic = time()
    frame_count = 0
    match_count = 0

    # Track last successful match position
    last_match_global = None

    while True:
        if camera.frame_q.empty():
            print(time(), " Frame Q Empty")
            sleep(1/camera_fps)
            if not camera.running.empty() and camera.process.is_alive():
                continue
            else:
                print("\nCamera closed. Press any key in CV window to close.")
                break

        fid, frame_x, frame_y, crop = camera.frame_q.get()
        print(f"\n{'='*70}")
        print(f"FRAME {fid}")
        print(f"{'='*70}")
        frame_count += 1

        # Apply noise
        crop = noise(crop, args.noise)

        # Apply blur
        crop = blur(crop, args.blur, (args.blur_filter, args.blur_filter))

        # Apply resize on input video
        if args.resize != 100:
            crop = cv2.resize(crop, new_size)

        w_array = []
        h_array = []
        thresh_list = []

        # Pass the template to data loader
        data = data_loader.load_template(crop)

        print(f"Current INS Point: {current_ins_point}")
        print(f"Search Region Offset: {data['crop_offset']}")

        # ========== VISUALIZE SEARCH REGION ==========
        if args.show_search_region and hasattr(data_loader, 'crop_offset'):
            # Get the crop boundaries in global coordinates
            offset_x, offset_y = data_loader.crop_offset
            region_w, region_h = data_loader.search_region_size

            # Calculate search region corners in scaled image coordinates
            search_x1 = offset_x // source_image_scale_factor
            search_y1 = offset_y // source_image_scale_factor
            search_x2 = (offset_x + region_w) // source_image_scale_factor
            search_y2 = (offset_y + region_h) // source_image_scale_factor

            # Draw search region rectangle on scaled_image (CYAN color)
            cv2.rectangle(scaled_image, 
                          (search_x1, search_y1),  # Top-left
                          (search_x2, search_y2),  # Bottom-right
                          (0, 255, 255),           # Cyan color (BGR)
                          3)                        # Thickness

            # Draw INS point marker (GREEN circle)
            ins_x_scaled = current_ins_point[0] // source_image_scale_factor
            ins_y_scaled = current_ins_point[1] // source_image_scale_factor
            cv2.circle(scaled_image, 
                       (ins_x_scaled, ins_y_scaled), 
                       7,                   # Radius
                       (0, 255, 0),         # Green color (BGR)
                       -1)                  # Filled circle

            # Add text label for search region
            cv2.putText(scaled_image, 
                        f"Search Region {fid}", 
                        (search_x1, search_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 255), 
                        2)

        # Run QATM matching on CROPPED search region
        score = run_one_sample_2(model, template=data['template'], image=data['image'])
        scores = np.squeeze(np.array([score]), axis=1)

        w_array.append(data['template_w'])
        w_array = np.array(w_array)
        h_array.append(data['template_h'])
        h_array = np.array(h_array)
        thresh_list.append(data['thresh'])

        # Apply NMS and threshold on the results of QATM
        mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=True)

        if len(mb_indices) > 0:
            match_count += 1

            # Map local coordinates to global coordinates
            local_x, local_y = mb_boxes[0][0][0], mb_boxes[0][0][1]
            global_x, global_y = data_loader.map_to_global_coords(local_x, local_y)

            print(f"\n{'='*70}")
            print(f"✓ MATCH FOUND! (Match #{match_count})")
            print(f"{'='*70}")
            print(f"  Local coords (search region):  ({local_x}, {local_y})")
            print(f"  Global coords (full image):    ({global_x}, {global_y})")

            # Store last successful match
            last_match_global = (global_x, global_y)

            # ========== DYNAMIC UPDATE: Update INS point for next frame ==========
            if args.dynamic_update:
                # The detected match becomes the NEW INS point for the next search
                new_ins_point = (int(global_x), int(global_y))

                print(f"\n  PROPAGATING SEARCH REGION:")
                print(f"    Old INS: {current_ins_point}")
                print(f"    New INS: {new_ins_point}")
                print(f"    Displacement: ({new_ins_point[0]-current_ins_point[0]}, {new_ins_point[1]-current_ins_point[1]})")

                # Update the search region for the next frame
                data_loader.set_ins_point(
                    ins_point=new_ins_point,
                    search_region_size=(args.search_region_width, args.search_region_height)
                )

                current_ins_point = new_ins_point
                print(f"  ✓ Search region updated for next frame")
            print(f"{'='*70}")

            # For visualization
            vis_boxes = mb_boxes.copy()
            vis_boxes[0][0][0] = global_x // source_image_scale_factor
            vis_boxes[0][0][1] = global_y // source_image_scale_factor

            d_img = plot_result(scaled_image, vis_boxes[0][None, :, :], 
                                text=f"{fid}", text_loc=(frame_x-20, frame_y-20))
            scaled_image = d_img
        else:
            print(f"\n✗ NO MATCH FOUND")
            print(f"  INS point unchanged: {current_ins_point}")
            print(f"  Search will continue in the same region (prevents drift)")
            print(f"{'='*70}")
            # Note: INS point is NOT updated if no match is found

        # Display results
        cv2.imshow("result", cv2.resize(d_img, (d_img.shape[1] // source_image_scale_factor,
                                                  d_img.shape[0] // source_image_scale_factor)))
        cv2.imshow("v-camera", crop)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        toc = time()
        print(f"Processing time: {toc - tic:.3f}s")
        tic = toc

    print("\n" + "="*70)
    print("PROCESSING COMPLETE - STATISTICS")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Total matches found:    {match_count}")
    print(f"Match rate:             {match_count/frame_count*100:.1f}%")
    if last_match_global:
        print(f"Final match position:   {last_match_global}")
    print(f"Search region size:     {args.search_region_width}x{args.search_region_height}")
    print("="*70)

    cv2.waitKey()
    cv2.destroyAllWindows()
