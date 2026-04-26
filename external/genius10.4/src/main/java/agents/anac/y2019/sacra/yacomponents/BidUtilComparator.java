package agents.anac.y2019.sacra.yacomponents;

import java.util.Comparator;

public class BidUtilComparator implements Comparator<BidUtility> {
    @Override
    public int compare(BidUtility b1, BidUtility b2) {
        return Double.compare(b1.getUtil(), b2.getUtil());
    }
}
