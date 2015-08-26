/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/

#include "highlyreliablemarkers.h"
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "board.h"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Invalid number of arguments" << endl;
        cerr
            << "Usage: dictionary.yml outputboard.yml outputimage.png height width [chromatic=0] [outdictionary.yml] \n \
      dictionary.yml: input dictionary from where markers are taken to create the board \n \
      outputboard.yml: output board configuration file in aruco format \n \
      outputimage.png: output image for the created board \n \
      height: height of the board (num of markers) \n \
      width: width of the board (num of markers) \n \
      chromatic: 0 for black&white markers, 1 for green&blue chromatic markers \n \
      outdictionary.yml: output dictionary with only the markers included in the board" << endl;
        exit(-1);
    }

    // read parameters
    std::string dictionaryfile = argv[1];
    std::string outboard = argv[2];
    std::string outimg = argv[3];
    cv::Size gridSize;
    gridSize.height = atoi(argv[4]);
    gridSize.width = atoi(argv[5]);
    bool chromatic = false;
    if (argc >= 7)
        chromatic = (argv[6][0] == '1');

    aruco::Dictionary D;
    D.fromFile(dictionaryfile);
    if (D.size() == 0) {
        std::cerr << "Error: Dictionary is empty" << std::endl;
        exit(-1);
    }

    aruco::BoardConfiguration BC;
    cv::Mat tableImage = aruco::HighlyReliableMarkers::createBoardImage(gridSize, D, BC, chromatic);


    BC.saveToFile(outboard); // save board configuration

    if (argc >= 8) {
        aruco::Dictionary outD;
        outD.insert(outD.begin(), D.begin(), D.begin() + gridSize.area());
        outD.toFile(argv[7]); // save new dictionary just with the used markers, if desired
    }


    cv::imshow("Board", tableImage);
    cv::waitKey(0);

    cv::imwrite(outimg, tableImage); // save output image
}
