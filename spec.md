
I'm aiming to train a ML model to predict where a dart player is aiming on the board based on their gaze behaviour. I've written a script to download a dart game video from youtube. There are some debug images in the debug folder. 

What is the best way to approach this problem? Is there some smart way of taking the video and creating clean data from it?

I can get the corresponding scores from the match to cross reference but to sync up maybe also getting the scores from the score pannel would be usful. 

Abstract from paper:

Gaze behaviour and arm movements of skilled dart players (N = 5) were recorded as they threw an equal
number of hit and misses to the centre of a regulation dart board. Quiet eye (QE) was defined as final
fixation on the target, with onset prior to the extension of the arm and offset when QE deviated off the
target. Accuracy was affected by the temporal offset of QE relative to the phases of the arm movement.
During hits, QE onset occurred during late alignment and offset during early flexion resulting in the gaze
being off the target for 550 ms. During misses, QE offset occurred during mid-alignment, resulting in the
gaze being off the target for 1167 ms. The results highlight the importance of the temporal control of QE
relative to the movements of arm. Quiet eye information gained too soon or too late did not lead to the same
level of accuracy as that obtained optimally, just prior to the final movement being initiated.