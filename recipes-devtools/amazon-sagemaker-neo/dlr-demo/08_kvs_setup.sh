source ./project_config.sh

cat > /etc/default/kvs << EOF
AWS_IOT_CORE_THING_NAME=$THING_NAME
AWS_IOT_CORE_CREDENTIAL_ENDPOINT=$IOT_CRED_ENDPOINT
AWS_IOT_CORE_ROLE_ALIAS=$ROLE_ALIAS_NAME
AWS_KVS_LOG_LEVEL=2
DEBUG_LOG_SDP=TRUE
AWS_DEFAULT_REGION=us-east-2                                  
AWS_KVS_CACERT_PATH=/kvs/certs/cert.pem                            
AWS_IOT_CORE_CERT=/greengrass/v2/device.pem.crt               
AWS_IOT_CORE_PRIVATE_KEY=/greengrass/v2/private.pem.key    
EOF

cat > /lib/systemd/system/kvs.service << EOF
[Unit]                                                                                                                                                                                        
Description=greengrass gst kvs demo                                                                                                                                                           
Documentation=                                                                                                                                                                                
After=network.target greengrass.service                                                                                                                                                       
                                                                                                                                                                                              
[Service]                                                                                                                                                                                     
EnvironmentFile=-/etc/default/kvs                                                                                                                                                             
PIDFile=/run/kvs/kvs.pid                                                                                                                                                                      
ExecStart=/kvs/kvsWebrtcClientMasterGstSample \$AWS_IOT_CORE_THING_NAME video-only testsrc
Restart=always                                                                                                                                                                                
StandardOutput=file:/var/log/kvs.log
StandardError=file:/var/log/kvs.log
Type=simple                                                                                                                                                                                   
RuntimeDirectory=kvs                                                                                                                                                                          
RuntimeDirectoryMode=0755                                                                                                                                                                     
                                                                                                                                                                                              
[Install]                                                                                                                                                                                     
WantedBy=multi-user.target                                                                                                                                                                    
Alias=kvs.service   
EOF

cp -r kvs /
systemctl daemon-reload
systemctl enable kvs
systemctl restart kvs


aws iot update-thing --region $AWS_REGION --thing-name $THING_NAME --attribute-payload '{"attributes":{"lat":"30.238523","lng":"-97.866010"}, "merge":true}'

