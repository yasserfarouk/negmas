package agents.anac.y2015.DrageKnight.etc;

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
		df = utilitySpace.getDiscountFactor();		// 割引効用を取得してる？
		rv = utilitySpace.getReservationValue();	// 留保価格を取得してる？
	}
	
	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			double offeredBidUtil = utilitySpace.getUtility(offeredBid);
			
			double ff = 1.0;							//割引効用を考慮した均一化関数
			if(df <= 0.75 && df > 0.5) {				// 割引効用が0.75ならこれ
				ff = 1 - (0.25 * time);
			} else if(df <= 0.5 && df > 0.25) {			// 割引効用が0.5ならこれ
				ff = 1 - (0.5 * time);
			} else if(df <= 0.25){						// 割引効用が0.25ならこれ
				ff = 1 - (0.75 * time);
			}
			
			if((offeredBidUtil * ff) >= getThreshold(time)){	// threshold < 効用*ff ならAccept 
				return true;
			} else {
				return false;
			}
		} catch (Exception e) {
			System.out.println("受容判定に失敗しました");
			e.printStackTrace();
			return false;
		}
	}
	
	// 交渉終了判定
	public boolean selectEndNegotiation(double time) {
		//メモ
		//ここ最近の向こうの提案がクソすぎる　かつ　留保価格が良ければEndNegoもあり？
		return false;
	}
	
	// メモ
	//割引効用が0.75以上と未満でthresholdを変えたい
	
	// 相手の提案に対する閾値を返す
	public double getThreshold(double time) {
		
		double e = 0.05;							// 通常のthresholdはこのeを使って計算する
		if(df <= 0.75 && df > 0.5) {				// 割引効用が0.75ならこれ
			e = 0.15;
		} else if(df <= 0.5 && df > 0.25) {			// 割引効用が0.5ならこれ
			e = 0.3;
		} else if(df <= 0.25){						// 割引効用が0.25ならこれ
			e = 1.5;
		}
		
		double threshold = 0.9 - Math.pow(time, 1/e);			//いい感じに妥協していくけど、相手の提案にはちょっと厳しい
		
		if(threshold < 0.6 && df == 1){
			threshold = 0.6;		//時間ギリギリまで、0.6未満ではAcceptしない
		} else if (threshold < 0.5 && df <= 0.75 && df > 0.5){
			threshold = 0.5;		//時間ギリギリまで、0.5未満ではAcceptしない
		} else if (threshold < 0.4 && df <= 0.5) {
			threshold = 0.4;
		}

		/* 交渉戦略に基づきthreshold(t)を設計する */
		/* negotiatingInfoから提案履歴の統計情報を取得できるので使っても良い */
		// 例：
		ArrayList<Object> opponents =  negotiatingInfo.getOpponents();
		for(Object sender:opponents){
			double m = negotiatingInfo.getAverage(sender);				//平均
			double v = negotiatingInfo.getVariancer(sender);			//分散
			double sd = negotiatingInfo.getStandardDeviation(sender);	//標準偏差
		}
		
		return threshold;
	}
	
	// 此方の提案に対する閾値を返す
		public double getThreshold2(double time) {
			double e = 0.08;
			if(df <= 0.75 && df > 0.5) {				// 割引効用が0.75ならこれ
				e = 0.2;
			} else if(df <= 0.5 && df > 0.25) {			// 割引効用が0.5ならこれ
				e = 0.35;
			} else if(df <= 0.25){						// 割引効用が0.25ならこれ
				e = 1.5;
			}
			
			double threshold = 1.0 - Math.pow(time, 1/e);					//いい感じに妥協していくけど
			
			if(df > 0.75){
				if(time > 0.8){
					threshold = 0.8;
				}
				if(time > 0.9){
					threshold = 0.6;
				}						//提案の時はちょっとだけ強気
			}
			
			// 全体的に強気思考だけど、強気同士でもなんとかなる？

			/* 交渉戦略に基づきthreshold(t)を設計する */
			/* negotiatingInfoから提案履歴の統計情報を取得できるので使っても良い */
			// 例：
			ArrayList<Object> opponents =  negotiatingInfo.getOpponents();
			for(Object sender:opponents){
				double m = negotiatingInfo.getAverage(sender);				//平均
				double v = negotiatingInfo.getVariancer(sender);			//分散
				double sd = negotiatingInfo.getStandardDeviation(sender);	//標準偏差
			}
			
			return threshold;
		}
}
