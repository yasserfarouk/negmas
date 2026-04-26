package agents.anac.y2015.AgentX;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class strategy {
	private AdditiveUtilitySpace utilitySpace;
	private negotiatingInfo negotiatingInfo;
	
	private double df = 0.0; // 割引係数
	private double rv = 0.0; // 留保価格

	public strategy(AdditiveUtilitySpace utilitySpace, negotiatingInfo negotiatingInfo) {		
		this.utilitySpace = utilitySpace;
		this.negotiatingInfo = negotiatingInfo;
		df = utilitySpace.getDiscountFactor();
		rv = utilitySpace.getReservationValue();
	}
	
	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			double offeredBidUtil = utilitySpace.getUtility(offeredBid);
			if(offeredBidUtil >= getThreshold(time)){ return true; }
			else{ return false; }
		} catch (Exception e) {
			System.out.println("受容判定に失敗しました");
			e.printStackTrace();
			return false;
		}
	}
	
	// 交渉終了判定
	public boolean selectEndNegotiation(double time) {
		return false;
	}
	
	// 閾値を返す
	public double getThreshold(double time) {
		double threshold = 1.0;
		double mi = 0.0;
		double ave = 0.0;
		double extra = 0.0;
		
		/* 交渉戦略に基づきthreshold(t)を設計する */
		/* negotiatingInfoから提案履歴の統計情報を取得できるので使っても良い */
		// 例：
		ArrayList<Object> opponents =  negotiatingInfo.getOpponents();
		//double sum = 0;
		//double sum_round = 0;
		//double mall = 0;
		
		for(Object sender:opponents){
			if ((negotiatingInfo.getPartnerBidNum(sender) % 10) == 0)
			{
				ave = 0;
				extra = 0;
			}			
			//sum = sum + negotiatingInfo.getAverage(sender) * negotiatingInfo.getPartnerBidNum(sender);
			//sum_round = sum_round + negotiatingInfo.getPartnerBidNum(sender);
			
			double m = negotiatingInfo.getAverage(sender);		
			// double v = negotiatingInfo.getVariancer(sender);
			double sd = negotiatingInfo.getStandardDeviation(sender);
            ave = m;
            extra = sd;
		}
		
		//ave = sum / sum_round;
		//System.out.println(ave);
		
        mi = 1 - Math.pow(time, 0.382);
        
        if(ave >= 0.618)
            threshold = ave + extra;
        else {
            threshold = mi;
        }
        return threshold;
    }
}
