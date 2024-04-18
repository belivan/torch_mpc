#ifndef EUCLIDEAN_DISTANCE_TO_MULTI_GOAL_IS_INCLUDED
#define EUCLIDEAN_DISTANCE_TO_MULTI_GOAL_IS_INCLUDED

#include "euclidean_distance_to_goal.h"

class EuclideanDistanceToMultiGoal : public EuclideanDistanceToGoal
{
    // Simple terminal cost that computes final-state distance to goal

    EuclideanDistanceToMultiGoal(){};
    ~EuclideanDistanceToMultiGoal() = default;
}