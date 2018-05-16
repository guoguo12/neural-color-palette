Crayon colors often have interesting names: official [Crayola colors](https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors) include "Inchworm", "Shamrock", "Pink Flamingo", and "Eggplant".
I used a neural network trained on ImageNet to generate better names for the [X11 colors](https://en.wikipedia.org/wiki/X11_color_names).

Here are the results:

![results](https://raw.githubusercontent.com/guoguo12/neural-crayon-pack/master/result.png)

The large text over each color is the predicted class, the number below it is the prediction probability, and the small text above the color is the color's official X11 name.

(The data for the X11 colors comes from the official X source code [here](https://cgit.freedesktop.org/xorg/app/rgb/tree/rgb.txt).)
