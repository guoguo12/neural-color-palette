# Neural Color Palette

Colors are often named after objects: official [Crayola colors](https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors) include "Inchworm", "Shamrock", "Pink Flamingo", and "Eggplant".
So I fed solid colors to a neural network (ResNet50 trained on ImageNet) to generate better names for the [X11 colors](https://en.wikipedia.org/wiki/X11_color_names).

Here are the results:

![](https://raw.githubusercontent.com/guoguo12/neural-crayon-pack/master/grid.png)

For each color, the large text shows the predicted class, the number below it is the prediction probability, and the small text above the color is the color's official X11 name.

There are 19 unique predicted color names, shown here with their corresponding maximum-posterior colors:

![](https://raw.githubusercontent.com/guoguo12/neural-crayon-pack/master/uniques.png)

## Acknowledgements

The names and values for the X11 colors come from the [X source code](https://cgit.freedesktop.org/xorg/app/rgb/tree/rgb.txt).
