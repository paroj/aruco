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

#ifndef HIGHLYRELIABLEMARKERS_H
#define HIGHLYRELIABLEMARKERS_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "aruco_export.h"
#include "board.h"

namespace aruco {

/**
 * This class represent the internal code of a marker
 * It does not include marker borders
 *
 */
class ARUCO_EXPORTS MarkerCode {
public:
    /**
     * Constructor, receive dimension of marker
     */
    MarkerCode(unsigned int n = 0);

    /**
     * Copy Constructor
     */
    MarkerCode(const MarkerCode& MC);

    /**
     * Get id of a specific rotation as the number obtaiend from the concatenation of all the bits
     */
    unsigned int getId(unsigned int rot = 0) const { return _ids[rot]; }

    /**
     * Get a bit value in a specific rotation.
     * The marker is refered as a unidimensional string of bits, i.e. pos=y*n+x
     */
    bool get(unsigned int pos, unsigned int rot = 0) const { return _bits[rot][pos]; }

    /**
     * Get the string of bits for a specific rotation
     */
    const std::vector<bool>& getRotation(unsigned int rot) const { return _bits[rot]; }

    /**
     * Set the value of all bits from a 2d bit matrix
     * This method assure consistency of the marker code:
     * - The rest of rotations are updated automatically when performing a modification
     * - The id values in all rotations are automatically updated too
     */
    void set(const cv::Mat& code);

    /**
     * Return the full size of the marker (n*n)
     */
    unsigned int size() const { return n() * n(); }

    /**
     * Return the value of marker dimension (n)
     */
    unsigned int n() const { return _n; }

    /**
     * Return the self distance S(m) of the marker (Equation 8)
     * Assign to minRot the rotation of minimun hamming distance
     */
    unsigned int selfDistance(unsigned int& minRot) const;

    /**
     * Return the self distance S(m) of the marker (Equation 8)
     * Same method as selfDistance(uint &minRot), except this doesnt return minRot value.
     */
    unsigned int selfDistance() const {
        unsigned int minRot;
        return selfDistance(minRot);
    }

    /**
     * Return the rotation invariant distance to another marker, D(m1, m2) (Equation 6)
     * Assign to minRot the rotation of minimun hamming distance. The rotation refers to the marker passed
     * as parameter, m
     */
    unsigned int distance(const MarkerCode& m, unsigned int& minRot) const;

    /**
     * Return the rotation invariant distance to another marker, D(m1, m2) (Equation 6)
     * Same method as distance(MarkerCode m, uint &minRot), except this doesnt return minRot value.
     */
    unsigned int distance(const MarkerCode& m) const {
        unsigned int minRot;
        return distance(m, minRot);
    }

    /**
     * Read marker bits from a string of "0"s and "1"s
     */
    void fromString(std::string s);

    /**
     * Convert marker to a string of "0"s and "1"s
     */
    std::string toString() const;

    /**
     * Convert marker to a cv::Mat image of (pixSize x pixSize) pixels
     * It adds a black border of one cell size
     */
    cv::Mat getImg(unsigned int pixSize) const;

private:
    unsigned int _ids[4];       // ids in the four rotations
    std::vector<bool> _bits[4]; // bit strings in the four rotations
    unsigned int _n;            // marker dimension
};

/**
 * This class represent a marker dictionary as a vector of MarkerCodes
 *
 *
 */
class ARUCO_EXPORTS Dictionary : public std::vector<MarkerCode> {
public:
    /**
     * Read dictionary from a .yml opencv file
     */
    bool fromFile(std::string filename);

    /**
     * Write dictionary to a .yml opencv file
     */
    bool toFile(std::string filename);

    /**
     * Return the distance of a marker to the dictionary, D(m,D) (Equation 7)
     * Assign to minMarker the marker index in the dictionary with minimun distance to m
     * Assign to minRot the rotation of minimun hamming distance. The rotation refers to the marker passed
     * as parameter, m
     */
    unsigned int distance(const MarkerCode& m, unsigned int& minMarker, unsigned int& minRot);

    /**
     * Return the distance of a marker to the dictionary, D(m,D) (Equation 7)
     * Same method as distance(MarkerCode m, uint &minMarker, uint &minRot), except this doesnt return
     * minMarker and minRot values.
     */
    unsigned int distance(const MarkerCode& m) {
        unsigned int minMarker, minRot;
        return distance(m, minMarker, minRot);
    }

    /**
     * Calculate the minimun distance between the markers in the dictionary (Equation 9)
     */
    unsigned int minimunDistance();

    int tau0;
};

/**
 * Highly Reliable Marker Detector Class
 *
 *
 */
class ARUCO_EXPORTS HighlyReliableMarkers {
public:
    /**
     * Load the dictionary that will be detected or read it directly from file
     */
    // correctionDistance [0,1] 0: totalmente restrictivo, 1 mas flexible
    static bool loadDictionary(Dictionary D, float correctionDistance = 1);
    static bool loadDictionary(std::string filename, float correctionDistance = 1);
    static Dictionary& getDictionary() { return _D; }


    /**
     * create marker dictionary
     * @param dictSize number of markers to add to the dictionary
     * @param n marker size
     * @return marker Dictionary
     */
    static Dictionary createDicitionary(size_t dictSize, size_t n);

    /**
     * Detect marker in a canonical image. Perform detection and error correction
     * Return marker id in 0 rotation, or -1 if not found
     * Assign the detected rotation of the marker to nRotation
     */
    static int detect(const cv::Mat& in, int& nRotations);

    /**Creates a printable image of a board
     * @param gridSize grid layout (numer of sqaures in x and Y)
     * @param D input dictionary from where markers are taken to create the board
     * @param BC output
     * @param chromatic true for green&blue chromatic markers
     */
    static cv::Mat createBoardImage(cv::Size gridSize, const Dictionary& D, BoardConfiguration& BC,
                                    bool chromatic = false);
private:
    /**
    * Balanced Binary Tree for a marker dictionary
    *
    */
    class BalancedBinaryTree {

    public:
        /**
        * Create the tree for dictionary D
        */
        void loadDictionary(Dictionary& D);

        /**
        * Search a id in the dictionary. Return true if found, false otherwise.
        */
        bool findId(unsigned int id, unsigned int& orgPos);

    private:
        std::vector<std::pair<unsigned int, unsigned int> > _orderD; // dictionary sorted by id,
                                                                     // first element is the id,
        // second element is the position in original D
        std::vector<std::pair<int, int> >
            _binaryTree;    // binary tree itself (as a vector), each element is a node of the tree
                            // first element indicate the position in _binaryTree of the lower child
                            // second element is the position in _binaryTree of the higher child
                            // -1 value indicates no lower or higher child
        unsigned int _root; // position in _binaryTree of the root node of the tree
    };

    static Dictionary _D; // loaded dictionary
    static BalancedBinaryTree _binaryTree;
    // marker dimension, marker dimension with borders, maximunCorrectionDistance
    static unsigned int _n;
    static unsigned int _ncellsBorder;
    static unsigned int _correctionDistance;
};
}

#endif // HIGHLYRELIABLEMARKERS_H
