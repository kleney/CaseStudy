#!/usr/bin/env bash

# ==============================================================
# Downloads all linked PDF files matching a given pattern from
# https://www.iata.org/en/publications/annual-review/
# and renames them in a common filename format.
# Author: Katharine Leney, April 2025
# ==============================================================

# Set paths and search patterns
sourcepath="https://www.iata.org/en/publications/annual-review/"
pattern="annual"
#pattern="iata-annual-review-2020"
destination="data/annual_reviews"

# Check that required commands are installed
# (should be by default as included in the conda environment)
required=(sed grep lynx wget); missing=()
for command in ${required[@]}; do
	hash $command 2>/dev/null || missing+=($command)
done
if (( ${#missing[@]} > 0 )); then
	echo "[FATAL] could not find command(s): ${missing[@]}. Exiting!"
	exit 1
fi

# Check that the output directory exists and create it if it doesn't
if [ ! -d "$destination" ]; then
    echo "Output directory -- $destination -- does not exist, making it now."
    mkdir "$destination"
fi

# Gets list of downloadable files and removes the first 6 characters (numbers in list)
echo "Getting list of files to download."
list=($(lynx -listonly -dump $sourcepath | grep $pattern | grep "pdf" | sed 's/^.\{6\}//'))

# Download each file to the specified directory.
# Skip if the file already exists.
for url in ${list[@]}; do

    # Extract filename from url
    #file=($(echo "$path" | awk -F'/' '{print $NF}'))
    file=($(echo "$url" | sed 's:.*/::'))
    echo "Checking if $file exists"
    
    if [ -f $destination/$file ]; then
	echo "$destination/$file already exists.  Skipping."
    else
	echo "Downloading $url"
	wget -q -P $destination --show-progress $url
    fi

done

# Do some very dumb, brute-force cleaning to remove the non-EN files.
# Would be better to just skip these in the first place...
rm $destination/*arabic.pdf
rm $destination/*chinese.pdf
rm $destination/*french.pdf
rm $destination/*spanish.pdf

echo "Renaming downloaded files to consistent format."
for oldname in $destination/*.pdf; do
  if [[ $oldname =~ ([0-9]{4}) ]]; then
    year="${BASH_REMATCH[1]}"
    newname="$destination/IATA-AnnualReview-${year}.pdf"
    mv "$oldname" "$newname"
    echo "Renamed '$oldname' to '$newname'"
  else
    echo "Could not find year in filename: $oldname"
  fi
done
