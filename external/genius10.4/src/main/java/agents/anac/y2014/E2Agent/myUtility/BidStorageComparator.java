package agents.anac.y2014.E2Agent.myUtility;

import java.util.Comparator;

public class BidStorageComparator implements Comparator {
    public int compare(Object s, Object t) {
        //               + (x > y)
        // compare x y = 0 (x = y)
        //               - (x < y)
        double diff = ((BidStorage) s).getUtility()
            - ((BidStorage) t).getUtility();
        int ret = 0;
        if(0 < diff) {
            ret = 1;
        } else if(diff < 0){
            ret = -1;
        }
        return ret;
    }
}
