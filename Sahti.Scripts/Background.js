const bg_video = document.querySelector('.bg-video');
const bg_video_source = document.querySelector('.bg_video_source');

let video_index = 1;

bg_video_source.src = "TheraBot.Media/Videos/video1.mp4";
bg_video.load();
bg_video.play();

bg_video.onended = function() {
    video_index++;
    if (video_index > 7) video_index = 1;
    bg_video_source.src = `TheraBot.Media/Videos/video${video_index}.mp4`;
    bg_video.load();
    bg_video.play();
};