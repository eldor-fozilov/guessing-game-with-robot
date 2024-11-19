from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bbox(image_path, bbox, class_name = None, confidence = None, normalized = True, save_path=None):

    if isinstance(image_path, Image.Image):
        image = image_path
    else:
        image = Image.open(image_path)

    image_width, image_height = image.size
    
    if normalized:
        
        x_min_pixel = bbox[0] * image_width
        y_min_pixel = bbox[1] * image_height
        x_max_pixel = bbox[2] * image_width
        y_max_pixel = bbox[3] * image_height

        bbox = [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]

    fig, ax = plt.subplots()

    ax.imshow(image)

    # Create a rectangle patch
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),  # (x_min, y_min)
        bbox[2] - bbox[0],   # width
        bbox[3] - bbox[1],   # height
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

    if class_name is not None and confidence is not None:
        label = f"{class_name} {confidence:.2f}"
        ax.text(
            bbox[0], bbox[1] - 10,  # (x_min, y_min) - slight offset above the bounding box
            label,
            color='white',
            fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=2)
        )

    ax.axis('off')

    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
