set xrange[0:132]
set style data histogram
set style histogram cluster gap 10
set title "kNN"
set ylabel "% hit"
set xlabel "K"
set style fill solid 1.0 border -1
set terminal png
set out 'result.png'
plot 'plot.data' with boxes