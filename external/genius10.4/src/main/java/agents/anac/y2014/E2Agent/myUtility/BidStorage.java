package agents.anac.y2014.E2Agent.myUtility;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.anac.y2011.TheNegotiator.BidsCollection;
import agents.anac.y2012.MetaAgent.agents.WinnerAgent.opponentOffers;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.timeline.Timeline;
import genius.core.utility.*;



public class BidStorage  implements Serializable {
    private double utility = -1;
    private Bid bid = null;
    private double time = -1;

    public BidStorage(Bid b, double u, double t) {
        bid = b;
        utility = u;
        time = t;
    }

    public Bid getBid() { return bid; }
    public double getUtility() { return utility; }
    public double getTime() { return time; }
    public String toString() {
        return bid.toString() + " Utility: " + utility + " Time: " + time;
    }

    // /**
    //  * å�ˆæ„�å½¢æˆ�ã�™ã‚‹ç¢ºçŽ‡ã‚’è¿”ã�™
    //  * @param u åŠ¹ç”¨
    //  * @param t1 çµŒé�Žæ™‚é–“0~1
    //  * @return å�ˆæ„�å½¢æˆ�ã�™ã‚‹ç¢ºçŽ‡
    //  * @throws uã�‹t1ã�Œä¸�æ­£ã�ªå€¤ã�®æ™‚
    //  */
    // double Paccept(double u, double t1) throws Exception {
    //     double t = t1 * t1 * t1;
    //     // æ­£è¦�åŒ–ã�™ã‚‹ã�¨uã�¯1.0ä»¥ä¸Šã�«ã�ªã‚‹æ™‚ã�Œã�‚ã‚‹ã�®ã�§1.05ã�¨ã�™ã‚‹
    //     if(u < 0 || u > 1.05) throw new Exception("utility "+u+" outside [0,1]");
    //     if(t < 0 || t > 1) throw new Exception("time "+t+" outside [0,1]");
    //     if(u > 1.0) u = 1;
    //     if(t == 0.5) return u;
    //     return (u - 2.0 * u * t + 2.0 * (-1.0 + t + Math.sqrt(sq(-1.0 + t) + u * (-1.0 + 2 * t))))/(-1. + 2*t);
    // }
}
