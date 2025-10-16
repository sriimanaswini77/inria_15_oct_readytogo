Here is the relevant code snippet from `inria_test_v2_DYNAMIC_WITH_VISUALIZATION.py` showing the calculation of the search region and the rectangle drawing part commented out as per your requirement:

# Pass the template to data loader
data = data_loader.load_template(crop)

# Commented out print statements to suppress output
# print(f"Current INS Point: {current_ins_point}")
# print(f"Search Region Offset: {data['crop_offset']}")

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
    
    # Drawing the search region rectangle is commented out so it's not shown:
    # cv2.rectangle(scaled_image,
    #     (search_x1, search_y1),  # Top-left corner
    #     (search_x2, search_y2),  # Bottom-right corner
    #     (0, 255, 255),           # Cyan color (BGR)
    #     3)                       # Thickness

    # Draw INS point marker (GREEN circle)
    ins_x_scaled = current_ins_point[0] // source_image_scale_factor
    ins_y_scaled = current_ins_point[1] // source_image_scale_factor
    cv2.circle(scaled_image,
        (ins_x_scaled, ins_y_scaled),
        7,  # Radius
        (0, 255, 0),  # Green color (BGR)
        -1)  # Filled circle

##########################################################

To remove the search region label text from the source image, comment out or remove this block in the visualization section of `inria_test_v2_DYNAMIC_WITH_VISUALIZATION.py`:

```python
cv2.putText(scaled_image,
    f"Search Region {fid}",
    (search_x1, search_y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 255),
    2)
```

Commented out, it will look like this:


# cv2.putText(scaled_image,
#     f"Search Region {fid}",
#     (search_x1, search_y1 - 10),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     0.5,
#     (0, 255, 255),
#     2)
```

This will keep the INS point marker but remove the text label of the search region from the display output.



