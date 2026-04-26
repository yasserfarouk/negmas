package agents.anac.y2018.agreeableagent2018;

/**
 * Created by Sahar Mirzayi
 * 5/23/2018
 * University of Tehran
 * Agent Lab.
 * Sahar.Mirzayi @ gmail.com
 */
public class Constants {
    public static double timeToConcede = 0.2;

    public static int smallDomainUpperBound = 1000;
    public static int midDomainUpperBound = 10000;

    public static double timeForUsingModelForSmallDomain = 0.2;  //0.3   //0.2
    public static double timeForUsingModelForMidDomain = 0.3;    //0.1   //0.3
    public static double timeForUsingModelForLargeDomain = 0.4;  //0.5   //0.4

    public static double neigExplorationDisFactor = 0.05;
    public static double concessionFactor = 0.1;
    // k \in [0, 1]. For k = 0 the agent starts with a bid of maximum utility
    public static double k = 0;

    public static double minimumUtility = 0.8;
}
