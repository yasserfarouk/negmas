package agents.anac.y2015.AgentW;

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
		
		int negonum = negotiatingInfo.getNegotiatorNum();
		
		double ou[] = new double[negonum];
		double avarage = 0;
		
		//System.out.println("nego :"+ou.length);
		
		if(negotiatingInfo.getOpponents().size()!=0){
			for(int i=0;i<negotiatingInfo.getOpponents().size();i++){
					
				//System.out.println(negotiatingInfo.getOpponents().get(i));
				//System.out.println("差："+negotiatingInfo.getgap(negotiatingInfo.getOpponents().get(i)));
								
				ou[i] = 4*negotiatingInfo.getgap(negotiatingInfo.getOpponents().get(i));
				ou[i] = ou[i] + 3*negotiatingInfo.getgapn(negotiatingInfo.getOpponents().get(i),1);
				ou[i] = ou[i] + 2*negotiatingInfo.getgapn(negotiatingInfo.getOpponents().get(i),2);
				ou[i] = ou[i] + negotiatingInfo.getgapn(negotiatingInfo.getOpponents().get(i),3);

				ou[i] = ou[i]/10;
				avarage = avarage + ou[i];
			}
			
			avarage = avarage / (negonum-1);
		}
		
		double e = 0.3;
		
		//double threshold = 1.0 - time;
		double threshold = 1.0 - Math.pow(time,1/e);
		
		if ((threshold - avarage > 0)&&(threshold - avarage < 1)){
			threshold = threshold - avarage;
		}
		
		/* 交渉戦略に基づきthreshold(t)を設計する */
		/* negotiatingInfoから提案履歴の統計情報を取得できるので使っても良い */
		/* 例：
		ArrayList<Object> opponents =  negotiatingInfo.getOpponents();
		for(Object sender:opponents){
			double m = negotiatingInfo.getAverage(sender);
			double v = negotiatingInfo.getVariancer(sender);
			double sd = negotiatingInfo.getStandardDeviation(sender);
		}
		*/
		
		return threshold;
	}

}
