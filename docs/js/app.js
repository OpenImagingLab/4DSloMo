$(document).ready(function() {
    // var editor = CodeMirror.fromTextArea(document.getElementById("bibtex"), {
    //     lineNumbers: false,
    //     lineWrapping: true,
    //     readOnly:true
    // });
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });

// var frameNumber = 0, // start video at frame 0
//     // lower numbers = faster playback
//     playbackConst = 500, 
//     // get page height from video duration
//     setHeight = document.getElementById("main"), 
//     // select video element         
//     vid = document.getElementById('v0'); 
//     // var vid = $('#v0')[0]; // jquery option

    
    

// // Use requestAnimationFrame for smooth playback
// function scrollPlay(){  
//   var frameNumber  = window.pageYOffset/playbackConst;
//   vid.currentTime  = frameNumber;
//   window.requestAnimationFrame(scrollPlay);
// console.log('scroll');
// }
    
// // dynamically set the page height according to video length
// vid.addEventListener('loadedmetadata', function() {
//   setHeight.style.height = Math.floor(vid.duration) * playbackConst + "px";
// });
    
    
//     window.requestAnimationFrame(scrollPlay);
});


function openTab(evt, tabName) {
    var i, x, tablinks;
    x = document.getElementsByClassName("bulma-content-tab");
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("bulma-tab");
    for (i = 0; i < x.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" is-active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " is-active";
}

// 视频播放控制函数
function playRecoloringVideo() {
    var video = document.getElementById('recoloring');
    var playBtn = document.getElementById('recoloringPlayBtn');
    
    if (video) {
        video.play();
        playBtn.style.display = 'none';
    }
}

// 检测autoplay是否工作
document.addEventListener('DOMContentLoaded', function() {
    var video = document.getElementById('recoloring');
    var playBtn = document.getElementById('recoloringPlayBtn');
    
    if (video && playBtn) {
        // 监听视频加载完成
        video.addEventListener('loadeddata', function() {
            // 尝试播放视频
            var playPromise = video.play();
            
            if (playPromise !== undefined) {
                playPromise.catch(function(error) {
                    // 如果自动播放被阻止，显示播放按钮
                    console.log('Autoplay was blocked, showing play button');
                    playBtn.style.display = 'block';
                });
            }
        });
        
        // 如果视频开始播放，隐藏播放按钮
        video.addEventListener('play', function() {
            playBtn.style.display = 'none';
        });
        
        // 如果视频暂停，显示播放按钮
        video.addEventListener('pause', function() {
            playBtn.style.display = 'block';
        });
    }
});