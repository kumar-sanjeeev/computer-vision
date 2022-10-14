from moviepy.editor import VideoFileClip
import lane_marking


clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(lane_marking.run_lane_detection) 
white_clip.write_videofile('project_video_output_test.mp4', audio = False)

