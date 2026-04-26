package agents.anac.y2014.E2Agent.myUtility;

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

public class AgentKStorategy {
    private Random randomnr = null;
    private IAgentKStorategyComponent component = null;

    public AgentKStorategy(Random r, IAgentKStorategyComponent com) {
        randomnr = r;
        component = com;
    }

    /**
     * ç›¸æ‰‹ã�Œå°†æ�¥çš„ã�«æ��æ¡ˆã�™ã‚‹ã�§ã�‚ã‚�ã�†Bidã�®æŽ¨æ¸¬æœ€å¤§åŠ¹ç”¨å€¤
     */
    public double emax(double myu, double var) {
        return  myu + (1 - myu) * d(var);
    }


    /**
     * ç›¸æ‰‹ã�®Bidã�®åºƒã�Œã‚Š
     * @param var ç›¸æ‰‹ã�®Bidã�«ã‚ˆã‚‹è‡ªèº«ã�®åŠ¹ç”¨ç©ºé–“ã�«ã�Šã�‘ã‚‹åŠ¹ç”¨å€¤ã�®åˆ†æ•£
     */
    private double d(double var) {
        return Math.sqrt(12.0 * var);
    }


    /**
     * æ��æ¡ˆã�™ã‚‹Bid
     * @param t æ™‚é–“
     * @param a å¦¥å�”ã�™ã‚‹é€Ÿåº¦ã‚’èª¿æ•´
     */
    private double target(double t, double myu, double var, double a) {
        return 1 - (1 - emax(myu, var)) * Math.pow(t, a);
    }

    /**
     * æ��æ¡ˆã�™ã‚‹Bidï¼ˆratioã‚’å°Žå…¥ï¼‰
     * @param t æ™‚é–“
     * @param var åˆ†æ•£
     * @param a å¦¥å�”ã�™ã‚‹é€Ÿåº¦ã‚’èª¿ç¯€
     */
    public double targetRatio(double t, double myu, double var, double a) {
        double r = ratio(t, myu, var, a, component.g(t));
        return r * (1 - (1 - emax(myu, var)) * Math.pow(t, a)) + (1 - r);
    }

    /**
     * ã�Šäº’ã�„ã�®è­²æ­©ã�®åº¦å�ˆã�„
     * @param t æ™‚é–“
     * @param a å¦¥å�”ã�™ã‚‹é€Ÿåº¦ã‚’èª¿æ•´
     * @param g æœ€ä½Žè­²æ­©åº¦å�ˆã�„
     */
    private double ratio(double t, double myu, double var, double a, double g) {
        double val = (d(var) + g) / (1 - target(t, myu, var, a));
        if (2  <= val) {
            val = 2;
        }
        return val;
    }

    /**
     * ç›¸æ‰‹ã�®Bidã�«å¯¾ã�—ã�¦å�ˆæ„�ã�™ã‚‹ã�‹ã�©ã�†ã�‹ã‚’åˆ¤æ–­ã�™ã‚‹æ™‚ã�«ç”¨ã�„ã‚‹ä¿‚æ•°
     */
    private double alpha(double myu, double tau) {
        return 1 + tau + 10 * myu - 2 * tau * myu;
    }

    /**
     * ç›¸æ‰‹ã�«Bidã�™ã‚‹æ™‚ã�«æ�ºã‚‰ã�Žã‚’ã‚‚ã�Ÿã�›ã‚‹ã�Ÿã‚�ã�®ä¿‚æ•°
     */
    private double beta(double myu, double tau) {
        return alpha(myu, tau) + randomnr.nextFloat() * tau - tau / 2.0;
    }

    /**
     * å�ˆæ„�ç¢ºçŽ‡
     * @param t æ™‚é–“
     * @param u ç›¸æ‰‹ã�®åŠ¹ç”¨å€¤
     * @param tau ä¿‚æ•°
     */
    public double pAccept(double t, double u, double myu, double var, double tau) {
        return (Math.pow(t, 5) / 5.0) +
            (u - emax(myu, var)) + (u - targetRatio(t, myu, var, alpha(myu, tau)));
    }


    /**
     * ç›¸æ‰‹ã�Œå¼·å›ºã�ªå ´å�ˆã�®æ­©ã�¿å¯„ã‚Šé–¢æ•°
     */
    private double approach(double t, double myu, double var, double a) {
        double ganma = -300 * t + 400;
        double delta = targetRatio(t, myu, var, a) - emax(myu, var);
        double epsilon = 1.0 / Math.pow(delta, 2);
        if (ganma <= epsilon) {
            epsilon = ganma;
        }
        return (delta * epsilon) / ganma;
    }


    /**
     * æœ€çµ‚çš„ã�ªBidã�®åŸºæº–é–¢æ•°
     */
    public double fintarget(double t, double myu, double var, double tau) {
        double b = beta(myu, tau);
        double em = emax(myu, var);
        double tar = targetRatio(t, myu, var, b);
        double ret = em;
        if (em < tar) {
            ret = tar - approach(t, myu, var, b);
        }
        return ret;
    }
}
