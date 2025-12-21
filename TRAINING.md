This project was seperated into distinct tasks. These tasks are:
- Draw card / Hold card to camera for ID
- Place card first slot
- Place card second slot
- Place card third slot
- Place card fourth slot
- Place card temp slot
- Pick card first slot
- Pick card second slot
- Pick card third slot
- Pick card fourth slot
- Pick card temp slot
- Discard
Each of these was trained as a discrete combination of neural network and preset paths, meaning that each could be trained in <25 episodes and <5K steps. 
One camera above, one camera to the side, and one camera on the arm. The above camera is also run a multiple virtual cameras within python for OpenCV and neural network interfacing.
