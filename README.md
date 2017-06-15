# Introduction

This reposiotry deals with 2D Simulataneously localization and mapping (SLAM). SLAM is the process which enables autonomous navigation of mobile robots or vehicles. It makes use of the kalman filter for mapping obstacles in its surroundings. Additionally we make use of GP regression for obstacle inference using a Gaussian kernel. For in detail information about mapped based localization and SLAM and the implementation of Gaussian processes, please refer the project report given as SLAM_GP.pdf.


## Files in this repository
* kernel&#46;py : Gaussian Process kernel used for gaussian process inference, created by Andy.
*  GPMap.ipynb : Python notebook used to create map of surrounding obstacles with Gaussian processes and for testing kernel&#46;py.
*  NOslam&#46;py : Python notebook, with animation showcasing how robot path will be without SLAM.
*  Simple Slam Modified.ipynb : Python notebook, in which SLAM has been implemented with Gaussian process inference. The robot moves on a circular path and maps the surrounding boundary as an obstacle.
*  Simple Slam with Path Planning.ipynb : Python notebook, in which we plan the robot path with implemenation with SLAM.

## Prerequisites For Running Notebook Animation:

To run the animations/movies, you will need to have FFmpeg installed on your PC. If you want to install FFmpeg in a windows machine do the following steps:

1. Download the FFmpeg program from FFmpeg download page. Link --> https://ffmpeg.org/download.html
2. On downloading, you should obtain a zip folder. Extract the zip folder and copy its contents.
3. On the drive where windows is installed (typically C:), create a new folder and rename it as FFMPEG. So if windows is installed on your C drive, the location of new FFMPEG folder must be in C:\FFMPEG.
4. Now copy and paste all the contents of the extracted zip folder downloaded from the ffmpeg website to the newly created FFMPEG folder.
5. After copying the contents, go to control panel, System and Security and then System. On the left side of your frame, select Advanced system settings.
6. Click the Environmental Variables button in the System Properties.
7. In the Variable value field, enter ;c:\ffmpeg\bin after anything that's already written there. If copied to a different drive, change the drive letter. Make sure you do not enter anything incorrectly, or else windows will have trouble booting.
8. If there is no path entry in the "User variables", click on the New button and create a new one. Enter PATH fot the variable name.
9. Following these instructions, FFMPEG should be installed. To check if FFMPEG is installed correctly, open command prompt and enter command "ffmpeg-version". If you receive "libstdc++ -6 is missing" error, you will need to install Visual C++ Restributable Package. If you still have trouble, visit this tutorial --> http://www.wikihow.com/Install-FFmpeg-on-Windows

To install FFMpeg on a linux system, open terminal enter the following command:

```sh
$ sudo apt-get install ffmpeg
```
