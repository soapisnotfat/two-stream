# RGB images
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
unzip ucf101_jpegs_256.zip

rm ucf101_jpegs_256.zip.001
rm ucf101_jpegs_256.zip.002
rm ucf101_jpegs_256.zip.003
rm  ucf101_jpegs_256.zip

# optical flow
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003

cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
unzip ucf101_tvl1_flow.zip

rm ucf101_tvl1_flow.zip.001
rm ucf101_tvl1_flow.zip.002
rm ucf101_tvl1_flow.zip.003
rm ucf101_tvl1_flow.zip