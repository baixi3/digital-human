var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    return pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .then(() => {
            return new Promise((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                    return;
                }

                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };

                pc.addEventListener('icegatheringstatechange', checkState);
            });
        })
        .then(() => {
            var offer = pc.localDescription;
            return fetch('/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('繁忙请重试');
            }
            return response.json();
        })
        .then((answer) => pc.setRemoteDescription(answer))
        .catch((e) => {
            alert(e.message || e);
        });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    pc = new RTCPeerConnection(config);

    pc.addEventListener('track', (evt) => {
        if (evt.track.kind === 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    setTimeout(() => {
        if (pc) {
            pc.close();
        }
    }, 500);
}
