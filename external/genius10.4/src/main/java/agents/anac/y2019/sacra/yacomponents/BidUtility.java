package agents.anac.y2019.sacra.yacomponents;

import genius.core.Bid;
import genius.core.utility.UtilitySpace;

public class BidUtility {
    protected Bid bid;

    protected double util;

    public BidUtility(Bid bid) {
        this(bid, 0.0);
    }

    public BidUtility(Bid bid, UtilitySpace utilitySpace) {
        this(bid, utilitySpace.getUtility(bid));
    }

    public BidUtility(Bid bid, double util) {
        this.bid = bid;
        this.util = util;
    }

    public Bid getBid() {
        return bid;
    }

    public double getUtil() {
        return util;
    }

    public void setUtil(double util) {
        this.util = util;
    }
}
