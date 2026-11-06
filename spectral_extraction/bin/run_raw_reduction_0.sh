echo "Starting raw reduction 0"
while read line; do
    $line
    echo "Tried..."
done </home/minerva/minerva-pipeline/spectral_extraction/bin/extract_000.txt
