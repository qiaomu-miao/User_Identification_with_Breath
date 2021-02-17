
var wavesurfer = WaveSurfer.create({
  container: '#waveform',
  waveColor: 'rgba(255,0,0,0.5)',
  progressColor: 'purple',
  height: 200,
  minimap: true,
  normalize: true
});

var wavesurfer_up = WaveSurfer.create({
  container: '#waveform2',
  waveColor: 'rgba(0,255,0,0.5)',
  progressColor: 'blue',
  height: 10,
  normalize: true
});

var GLOBAL_ACTIONS = {
    'play': function () {
        wavesurfer.playPause();
    },

    'play2': function () {
        wavesurfer_up.playPause();
    },

    'back': function () {
        wavesurfer.skipBackward();
    },

    'forth': function () {
        wavesurfer.skipForward();
    },

    'toggle-mute': function () {
        wavesurfer.toggleMute();
    }
};


// Bind actions to buttons and keypresses
document.addEventListener('DOMContentLoaded', function () {
    document.addEventListener('keydown', function (e) {
        var map = {
            32: 'play',       // space
            37: 'back',       // left
            39: 'forth'       // right
        };
        var action = map[e.keyCode];
        if (action in GLOBAL_ACTIONS) {
            if (document == e.target || document.body == e.target) {
                e.preventDefault();
            }
            GLOBAL_ACTIONS[action](e);
        }
    });

    [].forEach.call(document.querySelectorAll('[data-action]'), function (el) {
        el.addEventListener('click', function (e) {
            var action = e.currentTarget.dataset.action;
            if (action in GLOBAL_ACTIONS) {
                e.preventDefault();
                GLOBAL_ACTIONS[action](e);
            }
        });
    });
});


// Misc
document.addEventListener('DOMContentLoaded', function () {
    // Web Audio not supported
    if (!window.AudioContext && !window.webkitAudioContext) {
        var demo = document.querySelector('#demo');
        if (demo) {
            demo.innerHTML = '<img src="/example/screenshot.png" />';
        }
    }

});



wavesurfer.on('ready', function () {
// code that runs after wavesurfer is ready
console.log('Success');
});

function createCORSRequest(method, url){
  var xhttp=new XMLHttpRequest();
  if ("withCredentials" in xhttp){
    xhttp.open(method, url, true);
  } 
  else if (typeof XDomainRequest != "undefined"){
    xhttp= new XDomainRequest();
    xhttp.open(method, url);
  }
  else{
    xhttp=null;
  }
  return xhttp;
}


wavesurfer.on('ready',function(){
    wavesurfer.enableDragSelection({color:chooseColor()})
});

function chooseColor(type_flag){
    if (type_flag==1) return 'rgba(0, 10, 100, 0.1)'
    else return 'rgba(0, 100, 10, 0.1)'
}
wavesurfer.on('region-click', function (region, e) {
    e.stopPropagation();
    // Play on click, loop on shift click
    e.shiftKey ? region.playLoop() : region.play();
});
wavesurfer.on('region-click', editAnnotation);
wavesurfer.on('region-update-end', saveRegions);
wavesurfer.on('region-removed', saveRegions);
wavesurfer.on('region-in', showNote);

wavesurfer.on('region-play', function (region) {
    region.once('out', function () {
        wavesurfer.play(region.start);
        wavesurfer.pause();
    });
});




/* Timeline plugin */
wavesurfer.on('ready', function () {
    var timeline = Object.create(WaveSurfer.Timeline);
    timeline.init({
        wavesurfer: wavesurfer,
        container: "#wave-timeline",
        primaryLabelInterval: 10,
        secondaryLabelInterval: 10,
    });
});


/* Toggle play/pause buttons. */
var playButton = document.querySelector('#play');
var pauseButton = document.querySelector('#pause');
wavesurfer.on('play', function () {
    playButton.style.display = 'none';
    pauseButton.style.display = '';
});
wavesurfer.on('pause', function () {
    playButton.style.display = '';
    pauseButton.style.display = 'none';
});
var playButton2 = document.querySelector('#play2');
var pauseButton2 = document.querySelector('#pause2');
wavesurfer_up.on('play', function () {
    playButton2.style.display = 'none';
    pauseButton2.style.display = '';
});
wavesurfer_up.on('pause', function () {
    playButton2.style.display = '';
    pauseButton2.style.display = 'none';
});

    /**
 * Save annotations to localStorage.
 */
function saveRegions() {
    localStorage.regions = JSON.stringify(
        Object.keys(wavesurfer.regions.list).map(function (id) {
            var region = wavesurfer.regions.list[id];
            return {
                start: region.start,
                end: region.end,
                attributes: region.attributes,
                data: region.data
            };
        })
    );
}

/**
 * Edit annotation for a region.
 */
function editAnnotation (region) {
    var form = document.forms.edit;
    form.style.opacity = 1;
    form.elements.start.value = Math.round(region.start * 10) / 10,
    form.elements.end.value = Math.round(region.end * 10) / 10;
    form.elements.note.value = region.data.note || '';
    form.onsubmit = function (e) {
        e.preventDefault();
        region.update({
            start: form.elements.start.value,
            end: form.elements.end.value,
            data: {
                note: form.elements.note.value
            }
        });
        form.style.opacity = 0;
    };
    form.onreset = function () {
        form.style.opacity = 0;
        form.dataset.region = null;
    };
    form.dataset.region = region.id;
}


/**
 * Display annotation.
 */
function showNote (region) {
    if (!showNote.el) {
        showNote.el = document.querySelector('#subtitle');
    }
    showNote.el.textContent = region.data.note || 'untitled';
}

/**
 * Bind controls.
 */
GLOBAL_ACTIONS['delete-region'] = function () {
    var form = document.forms.edit;
    var regionId = form.dataset.region;
    if (regionId) {
        wavesurfer.regions.list[regionId].remove();
        form.reset();
    }
};

GLOBAL_ACTIONS['export'] = function () {
    window.open('data:application/json;charset=utf-8,' +
        encodeURIComponent(localStorage.regions));
};

var file_input=document.getElementById('uploadTrack');
file_input.addEventListener("change", loadNewAudio, false);

function loadNewAudio(){
    wavesurfer.loadBlob(file_input.files[0]);
}

function loadNewAudioUpload(){
    wavesurfer_up.load('http://localhost:7000/curr_audio.wav')
}

function upload_file(){
    var input=document.getElementById('uploadTrack')
    if ('files' in input){

        if (input.files.length == 0){
            txt="Please select files"
        }
        else{
            txt="Upload Successful!"
            wavesurfer.clearRegions()
            url='http://localhost:7000/getaudio'
            var formData = new FormData();
            var request = createCORSRequest('post',url);
            var content = input.files[0];
            // console.log(content);
            formData.append("file", content);
            formData.append("url",window.URL.createObjectURL(input.files[0]));
            formData.append("type",content.type);
            // //request.open("POST", url, true);
            if (request){
                request.open('post',url, true);
                request.send(formData);
                request.onload= function () {
                    //console.log(request.responseText)
                    obj= JSON.parse(request.responseText);  // gives a json object
                    var resp=obj.msg;
                    var output_person = obj.output_person
                    var output_prob = obj.output_probability
                    var total_time = obj.total_time

                    document.getElementsByName('output')[0].value= output_person;
                    document.getElementsByName('output')[1].value= output_prob;
                    document.getElementsByName('output')[2].value= total_time;


                    //wavesurfer.addRegion({id:'1',start:start_time,end:end_time, drag: true, color: "rgba(0, 10, 100, 0.1)"});
                    // document.getElementById("file_upload_text").innerHTML = resp;
                    // console.log(wavesurfer.regions.list);
                };
            }
        }   
    }
    else{
        txt="files not in input"
    }
    document.getElementById("demo").innerHTML = txt;
};


function classify_sample(filename){
    
    wavesurfer.clearRegions()
    url='http://localhost:7000/getsample'
    var formData = new FormData();
    var request = createCORSRequest('post',url);
    // var content = input.files[0];
    // console.log(content);
    // formData.append("file", content);
    formData.append("filename",filename);
    formData.append("type",'audio/wav');
    // //request.open("POST", url, true);
    if (request){
        request.open('post',url, true);
        request.send(formData);
        request.onload= function () {
            //console.log(request.responseText)
            obj= JSON.parse(request.responseText);  // gives a json object
            var resp=obj.msg;
            var output_person = obj.output_person
            var output_prob = obj.output_probability
            var total_time = obj.total_time
            
            document.getElementsByName('output')[0].value= output_person;
            document.getElementsByName('output')[1].value= output_prob;
            document.getElementsByName('output')[2].value= total_time;

        };
    }
    
};

function loadsample1(){
  // load file into wavesurfer
  wavesurfer.load('http://localhost:7000/sample_1.wav')
  
  var labelTag=document.getElementById( 'file-label' );
  labelTag.innerHTML = "Sample 1";
  // Call classifier 
  classify_sample('Qiaomu.wav')
};
function loadsample2(){
  // load file into wavesurfer
  wavesurfer.load('http://localhost:7000/sample_2.wav')

  var labelTag=document.getElementById( 'file-label' );
  labelTag.innerHTML = "Sample 2";

  // Call classifier 
  classify_sample('Qiaomu.wav')

};
function loadsample3(){
  // load file into wavesurfer
  wavesurfer.load('http://localhost:7000/sample_3.wav')

  var labelTag=document.getElementById( 'file-label' );
  labelTag.innerHTML = "Sample 3";

  // Call classifier 
  classify_sample('Qiaomu.wav')

};
function loadsample4(){
  // load file into wavesurfer
  wavesurfer.load('http://localhost:7000/sample_4.wav')
  var labelTag=document.getElementById( 'file-label' );
  labelTag.innerHTML = "Sample 4";

  // Call classifier 
  classify_sample('Qiaomu.wav')
};


