/*
 **********************************
 *
 * Copyright 2022 NXP
 *
 **********************************
 */

import React, { Component } from 'react';
import { FullscreenOutlined, EnvironmentOutlined, CloseCircleOutlined, VideoCameraOutlined } from '@ant-design/icons';

import GoogleMap from 'google-map-react';
import { Table } from 'antd';
import 'antd/dist/antd.css';
import './style.css';
import axios from 'axios';

//import { Player } from 'video-react';

import AWS from "aws-sdk";
import { Role, SignalingClient } from 'amazon-kinesis-video-streams-webrtc'


function uid() {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}


const Marker = ({ thingName, onClick, hoveredRowIndex }) => <div
onClick={onClick}
id={thingName} className="mark">
  {(hoveredRowIndex === thingName) ?
   <VideoCameraOutlined style={{fontSize: '39px', color: 'rgb(9 227 198)', zIndex: 10000, position: 'absolute'}} />
   : <VideoCameraOutlined style={{fontSize: '33px', color: '#d92424', zIndex: 999}} />

  }
</div>;

const InfoWindow = ({ src, thingName, onClickClose, onClickTop, hoveredRowIndex, opts }) =>
      <div
        onClick={onClickTop}
        style={(hoveredRowIndex === thingName) ?
               {marginTop:'-210px', float:'left', background: '#FFF', padding: '6px', width: "300px", height: "200px", zIndex: 10000} :
               {marginTop:'-210px', float:'left', background: '#FFF', padding: '6px', width: "300px", height: "200px", zIndex: 100}
        }
      id="info_win" >
  {/*
  <Player
    playsInline
    autoPlay={true}
    loop={true}
    src={src}
    fluid={false}
    width={"100%"}
  />
   */}
  <LiveFeedView formValues={opts} id={thingName}></LiveFeedView>

  <div className="bar">
    <span> {thingName} </span>
    <CloseCircleOutlined onClick={onClickClose}/>
    <FullscreenOutlined onClick={() => {
      var el = document.getElementById(thingName);
      if (el.requestFullscreen) {
          el.requestFullscreen();
      } else if (el.msRequestFullscreen) {
          el.msRequestFullscreen();
      } else if (el.mozRequestFullScreen) {
          el.mozRequestFullScreen();
      } else if (el.webkitRequestFullscreen) {
          el.webkitRequestFullscreen();
      }
      }
    }/>
  </div>
</div>

class LiveFeedView extends React.Component {
    constructor(props) {
        super(props);
        this.videoRef = React.createRef()
        this.viewer = {};
    }

    componentWillUnmount() {
        console.log('[VIEWER] Stopping viewer connection');
        if (this.viewer.signalingClient) {
            this.viewer.signalingClient.close();
            this.viewer.signalingClient = null;
        }

        if (this.viewer.peerConnection) {
            this.viewer.peerConnection.close();
            this.viewer.peerConnection = null;
        }

        if (this.viewer.remoteStream) {
            this.viewer.remoteStream.getTracks().forEach(track => track.stop());
            this.viewer.remoteStream = null;
          console.log("track stop")
        }

        if (this.viewer.peerConnectionStatsInterval) {
            clearInterval(this.viewer.peerConnectionStatsInterval);
            this.viewer.peerConnectionStatsInterval = null;
        }

        if (this.viewer.remoteView) {
            this.viewer.remoteView.srcObject = null;
        }

        if (this.viewer.dataChannel) {
            this.viewer.dataChannel = null;
        }
    }

    async componentDidMount() {
        // Create KVS client
        AWS.config.region = 'us-east-2'; // Region
        AWS.config.credentials = new AWS.CognitoIdentityCredentials({
          IdentityPoolId: 'us-east-2:xxxxxxxxxxxxxx',
          RoleArn: "arn:aws:iam::xxxxxxx:role/xxxxxxx",
          });
        const kinesisVideoClient = new AWS.KinesisVideo({
            region: this.props.formValues.region,
            //accessKeyId: this.props.formValues.accessKeyId,
            //secretAccessKey: this.props.formValues.secretAccessKey,
            //sessionToken: this.props.formValues.sessionToken,
            endpoint: this.props.formValues.endpoint,
        });

        // Get signaling channel ARN
        const describeSignalingChannelResponse = await kinesisVideoClient.describeSignalingChannel({ ChannelName: this.props.formValues.channelName}).promise();
        const channelARN = describeSignalingChannelResponse.ChannelInfo.ChannelARN;
        console.log('[VIEWER] Channel ARN: ', channelARN);

        // Get signaling channel endpoints
        const getSignalingChannelEndpointResponse = await kinesisVideoClient.getSignalingChannelEndpoint({ ChannelARN: channelARN,
                SingleMasterChannelEndpointConfiguration: {
                    Protocols: ['WSS', 'HTTPS'],
                    Role: Role.VIEWER,
                },
            }).promise();

        const endpointsByProtocol = getSignalingChannelEndpointResponse.ResourceEndpointList.reduce((endpoints, endpoint) => {
            endpoints[endpoint.Protocol] = endpoint.ResourceEndpoint;
            return endpoints;
        }, {});
        console.log('[VIEWER] Endpoints: ', endpointsByProtocol);

        const kinesisVideoSignalingChannelsClient = new AWS.KinesisVideoSignalingChannels({
            region: this.props.formValues.region,
            //accessKeyId: this.props.formValues.accessKeyId,
            //secretAccessKey: this.props.formValues.secretAccessKey,
            //sessionToken: this.props.formValues.sessionToken,
            endpoint: endpointsByProtocol.HTTPS,
        });

        // Get ICE server configuration
        const getIceServerConfigResponse = await kinesisVideoSignalingChannelsClient.getIceServerConfig({
                ChannelARN: channelARN,
            }).promise();

        const iceServers = [];
        iceServers.push({ urls: `stun:stun.kinesisvideo.${this.props.formValues.region}.amazonaws.com:443` });
            getIceServerConfigResponse.IceServerList.forEach(iceServer =>
                iceServers.push({
                    urls: iceServer.Uris,
                    username: iceServer.Username,
                    credential: iceServer.Password,
                }),
            );
        console.log('[VIEWER] ICE servers: ', iceServers);

        // Create Signaling Client
        this.viewer.signalingClient = new SignalingClient({
            channelARN,
          forceTURN: true,
            channelEndpoint: endpointsByProtocol.WSS,
            clientId: uid(),
            role: Role.VIEWER,
            region: this.props.formValues.region,
            //credentials: {
            //    accessKeyId: this.props.formValues.accessKeyId,
            //    secretAccessKey: this.props.formValues.secretAccessKey,
            //    sessionToken: this.props.formValues.sessionToken,
            //},
          credentials: AWS.config.credentials ,
        });

        const configuration = {
            iceServers,
            iceTransportPolicy: 'all',
        };
        this.viewer.peerConnection = new RTCPeerConnection(configuration);

        this.viewer.signalingClient.on('open', async () => {
            console.log('[VIEWER] Connected to signaling service');

            // Create an SDP offer to send to the master
            console.log('[VIEWER] Creating SDP offer');
            await this.viewer.peerConnection.setLocalDescription(
                await this.viewer.peerConnection.createOffer({
                    offerToReceiveAudio: true,
                    offerToReceiveVideo: true,
                }),
            );

            // When trickle ICE is enabled, send the offer now and then send ICE candidates as they are generated. Otherwise wait on the ICE candidates.
            console.log('[VIEWER] Sending SDP offer');
            this.viewer.signalingClient.sendSdpOffer(this.viewer.peerConnection.localDescription);
            console.log('[VIEWER] Generating ICE candidates');
        });

        this.viewer.signalingClient.on('sdpAnswer', async answer => {
            // Add the SDP answer to the peer connection
            console.log('[VIEWER] Received SDP answer');
            await this.viewer.peerConnection.setRemoteDescription(answer);
        });

        this.viewer.signalingClient.on('iceCandidate', candidate => {
            // Add the ICE candidate received from the MASTER to the peer connection
            console.log('[VIEWER] Received ICE candidate');
            this.viewer.peerConnection.addIceCandidate(candidate);
        });

        this.viewer.signalingClient.on('close', () => {
            console.log('[VIEWER] Disconnected from signaling channel');
        });

        this.viewer.signalingClient.on('error', error => {
            console.error('[VIEWER] Signaling client error: ', error);
        });

        // Send any ICE candidates to the other peer
        this.viewer.peerConnection.addEventListener('icecandidate', ({ candidate }) => {
            if (candidate) {
                console.log('[VIEWER] Generated ICE candidate');

                // When trickle ICE is enabled, send the ICE candidates as they are generated.
                console.log('[VIEWER] Sending ICE candidate');
                this.viewer.signalingClient.sendIceCandidate(candidate);
            } else {
                console.log('[VIEWER] All ICE candidates have been generated');
            }
        });

        // As remote tracks are received, add them to the remote view
        this.viewer.peerConnection.addEventListener('track', async (event) => {
            console.log('[VIEWER] Received remote track');
            this.viewer.remoteStream = event.streams[0];
            this.videoRef.current.srcObject = event.streams[0];
        });

        console.log('[VIEWER] Starting viewer connection');
        this.viewer.signalingClient.open();
    }

    render() {
        return (
          <div className="videoRTC">
            <video id={this.props.id} ref={this.videoRef} style={{width: '100%', position: 'relative' }} autoPlay playsInline />

          </div>
        )
    }
}

class CustomMarker extends React.Component{
  constructor(props){
    super(props)
    this.state = {
      showInfo: false,
      opts: {...this.props.awsConfig, channelName: this.props.thingName}
    }
    this.videoRef = React.createRef()

    console.log(this.props)
  }

  render(){
    return(
        <div>
          {this.state.showInfo && (
              <InfoWindow
                onClickClose={this.onClickMarkerClose.bind(this)}
                onClickTop={this.onClickMarkerTop.bind(this)}
                src = {this.props.video_url}
                thingName = {this.props.thingName}
                opts = {this.state.opts}
                {...this.props}
              >
              </InfoWindow>
          )}
          <Marker
            key={JSON.stringify(this.props.position)}
            {...this.props}
            onClick={this.onClickMarker.bind(this)}
          >
          </Marker>
        </div>
    )
  }

  onClickMarker(){
    this.setState({showInfo: !this.state.showInfo})
    console.log(this.props)
  }
  onClickMarkerClose(){
    this.setState({showInfo: false})
  }
  onClickMarkerTop(){
    var rowEL = document.getElementById("rowID_"+this.props.thingName)
    rowEL.click()
  }
}

//@controllable()
class SimpleMap extends Component {
  constructor(props){
    super(props);
    this.state = {
      isLoaded : false,
      center: [11, 22],
      zoom: 1,
      hoveredRowIndex: -1,
    };
    console.log(this.props)
  }

  componentDidMount() {
    console.log("Component did mount", this.state.center)
    this.getData()
  }

  componentDidUpdate() {
    console.log("Component did update", this.state.center)
  }

  getData(){
    const _this=this;
    axios.get('https://hcbj5osrr8.execute-api.us-east-1.amazonaws.com/v1/things',
              {withCredentials: false})
      .then(function (response) {
        _this.setState({
          devices:response.data["things"],
          isLoaded:true,
          awsConfig: {
            accessKeyId: "",
            secretAccessKey: "",
            sessionToken: "",
            region: "us-east-2"
  }
        });
      })
      .catch(function (error) {
        console.log(error);
        _this.setState({
          isLoaded:false,
          error:error
        });
      });
  }

  test(id, lat, lng){
    this.setState({center: [lat, lng]});
    this.setState({zoom: 12});
    this.setState({hoveredRowIndex: id});
  }

  render() {
    const columns = [
      {
        title: 'Thing Name',
        dataIndex: 'thingName',
        key: 'id',
        render: (id, record) => <a id={"rowID_"+id} onClick={this.test.bind(this, id, record['lat'], record['lng'])}> <EnvironmentOutlined title={ record['lat'] + ", " + record['lng']} /> { id } </a>
      },
    ]

    if (!this.state.isLoaded) {
      return (
          <div >
            <p>Loading ......</p>
          </div>
      )
    }

    const places = this.state.devices
          .map(place => {
            const {thingName, ...coords} = place;
            return (
                <CustomMarker
                   key={thingName}
                   hoveredRowIndex={this.state.hoveredRowIndex}
                   {...coords}
                   thingName={thingName}
                   awsConfig={this.state.awsConfig}
                   stylePtPos={this.state.hoveredRowIndex === thingName ?  {zIndex: 111} : {zIndex: 222}}
                />
            );
          });


    return (
        <div>
          <div style={{ height: '100vh', width: '80%', float: 'left'}}>
          <GoogleMap
            bootstrapURLKeys={{ key: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" }}
            center={this.state.center}
            zoom={this.state.zoom}
            options = {{disableDoubleClickZoom:false}}
            yesIWantToUseGoogleMapApiInternals
          >
            {places}
          </GoogleMap>
        </div>
        <div style={{ height: '100px', width: '20%', float: 'left'}} >
          <Table dataSource={this.state.devices} columns={columns} />
          </div>
        </div>
    );
  }
}

export default SimpleMap;
