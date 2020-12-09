# Using Principal Path to walk through music and visual art style spaces induced by CNN

This project aims at navigating the Music and Visual Art Spaces via the Principal Path algorithm (1).

1. 'Finding Prinicpal Paths in Data Space', M.J.Ferrarotti, W.Rocchia, S.Decherchi

## Downloads files (mandatory)

```
git clone https://github.com/erikagardini/Using-PP-to-walk-through-music-and-visual-art-style-spaces-induced-by-CNN.git
```

## Hot to use this code

You can install python requirements with

```
pip3 install -r requirements.txt
```

## Walking through Visual Art Space

You can compute the Principal Path with visual artworks using the following command

```
cd python
python3 main.py "images" <mode> <list-of-styles>
```
where the parameter **mode** allows to choose the start and end points and can be:
- 0: from the centroid of the first class specified to the centroid of the last class specified
- 1: selected visually by the user
- 2: from the most recent to the oldest visual artworks
  
and the parameter **list-of-styles** is the subset of styles you want to select. You can insert the numbers divided by a blank space remembering the following matching:
- 1: 'Early_Renaissance'
- 2: 'Naïve_Art_(Primitivism)',
- 3: 'Expressionism'
- 4: 'Magic_Realism'
- 5: 'Northern_Renaissance'
- 6: 'Rococo'
- 7: 'Ukiyo-e'
- 8: 'Art_Nouveau_(Modern)'
- 9: 'Pop_Art'
- 10: 'High_Renaissance'
- 11: 'Minimalism'
- 12: 'Mannerism_(Late_Renaissance)'
- 13: 'Art_Informel'
- 14: 'Neoclassicism'
- 15: 'Color_Field_Painting'
- 16: 'Symbolism'
- 17: 'Realism'
- 18: 'Romanticism'
- 19: 'Surrealism'
- 20: 'Cubism'
- 21: 'Impressionism'
- 22: 'Baroque'
- 23: 'Abstract_Expressionism'
- 24: 'Post-Impressionism'
- 25: 'Abstract_Art

### Example
You can compute the Principal Path selecting all the visual artworks belonging to the Baroque, the Neoclassicism, the Realism and the Expressionism, using as start and end points respectively the most recent visual artwork belonging to the Baroque and the oldest artwork belonging to the Expressionism, running the following command:

```
python3 main.py "images" 2 22 14 17 3 
```

The code produces the following output:
- [KNNpp.svg](results/images/mode=2_22-14-17-3/Recovered%20styles%20progression%20pp.svg): the labels of the nearest artwork for each waypoint obtained with the Principal Path algorithm (pp)
- [KNNtp.svg](results/images/mode=2_22-14-17-3/Recovered%20styles%20progression%20tp.svg): the labels of the nearest artwork for each waypoint obtained with the trivial path (tp)
- [paths.svg](results/images/mode=2_22-14-17-3_ok/perturbations/first_second/paths.svg): the 2D visualization of the Principal Path and the trivial path with t-SNE
- [pp_info.txt](results/images/mode=2_22-14-17-3_ok/perturbations/first_second/pp_info.txt): the information about the nearest artwork (style, author, name, date) for each waypoint obtained with the Principal Path algorithm (pp)
- [tp_info.txt](results/images/mode=2_22-14-17-3_ok/perturbations/first_second/tp_info.txt): the information about the nearest artwork (style, author, name, date) for each waypoint obtained with the trivial path (tp)

If you want to match the lines inside the \*\_info.txt files, you can collect the images in the [WikiArt website](https://www.wikiart.org) or you can download the full Wikipainting dataset from the [RASTA project's github](https://github.com/bnegreve/rasta) (2) executing the following command:
```
wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/rasta_models.tgz
tar xzvf rasta_models.tgz
```

2. 'Recognizing Art Style Automatically in painting with deep learning', A. Lecoutre, B. Negrevergne, F. Yger

##  Walking through Music Space

You can compute the Principal Path with music using the following command

```
python3 main.py "music" <mode> <list-of-music-genres>
```
where the parameter **mode** allows to choose the start and the end points and can be:
- 0: from the centroid of the first class specified to the centroid of the last class specified
- 1: selected visually by the user
  
and the parameter **list-of-genres** is the subset of genres you want to select. You can insert the numbers divided by a blank space remembering the following matching:
- 1: "classical"
- 2: "baroque"
- 3: "rock"
- 4: "opera"
- 5: "medieval"
- 6: "jazz"

### Example
You can compute the Principal Path selecting all the songs belonging to the Baroque, Jazz and Rock genres, visually selecting the start and the end points, running the following command:

```
python3 main.py "music" 1 2 6 3 
```

The code produces the following output (start point index = 21, end point index = 579):
- [KNNpp.svg](results/music/mode=1_2-6-3/Recovered%20styles%20progression%20pp.svg): the labels of the nearest song for each waypoint obtained with the Principal Path algorithm (pp)
- [KNNtp.svg](results/music/mode=1_2-6-3/Recovered%20styles%20progression%20tp.svg): the labels of the nearest song for each waypoint obtained with the trivial path (tp)
- [paths.svg](results/music/mode=1_2-6-3/paths.svg): the 2D visualization of the Principal Path and the trivial path with t-SNE
- [pp_info.txt](results/music/mode=1_2-6-3/pp_info.txt): the information about the nearest song (genre, author, name) for each waypoint obtained with the Principal Path algorithm (pp)
- [tp_info.txt](results/music/mode=1_2-6-3/tp_info.txt): the information about the nearest song (genre, author, name) for each waypoint obtained with the trivial path (tp)

If you want to match the lines inside the \*\_info.txt files, you can download the Magnatagatune dataset in [here](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).

**NB**: we insert inside the \*\_info.txt files the indexes of the start and the end points when we use mode=1 (in order to make our results reproducible).  

