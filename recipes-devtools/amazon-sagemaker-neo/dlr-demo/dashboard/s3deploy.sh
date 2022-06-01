#go get https://github.com/bep/s3deploy
#cp trailer_hd.mp4 build/static
s3deploy -bucket greengrass-cloud-demo.www -region aws-global -config s3deploy.yml  -public-access -source build
