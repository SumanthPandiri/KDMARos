# KDMARos
Developed during my REU in Summer 2023, this is my ROS2 implementation of my [KDMA code]([url](https://github.com/SumanthPandiri/KDMA)), tested on the TurtleBot4. We attached HTC Vive 2 Trackers to the TurtleBot and goal destinations. We then published location and orientation information using [this code]([url](https://github.com/ydhadix/vivetracker_ros)). We input this live data of neighbor locations and the goal location into the KDMA policy, and then take the outputs command velocities and publish them to the TurtleBot. 

Known issue: Nonfluid movements and imperfections in KDMA Differential Drive Policy

See KDMA Repo [here](https://github.com/SumanthPandiri/KDMA).

