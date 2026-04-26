package agents.anac.y2015.Atlas3.etc;

import agents.anac.y2015.Atlas3.Atlas3;
import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class strategy {
	private AdditiveUtilitySpace utilitySpace;
	private negotiatingInfo negotiatingInfo;
	
	private double df = 0.0; // 割引係数
	private double rv = 0.0; // 留保価格
	
	private double A11 = 0.0; // 効用値・・・A:強硬戦略，B:強硬戦略
	private double A12 = 0.0; // 効用値・・・A:強硬戦略，B:妥協戦略
	private double A21 = 0.0; // 効用値・・・A:妥協戦略，B:強硬戦略
	private double A22 = 0.0; // 効用値・・・A:妥協戦略，B:妥協戦略

	static private double TF = 1.0; // 最終提案フェーズの時刻
	static private double PF = 0.5; // 最終提案フェーズにおいて互いのが妥協戦略を選択した場合に，自身が相手よりも先に譲歩する確率
	
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
		// 閾値が留保価格を下回るとき交渉を放棄
		if (rv * Math.pow(df, time) >= getThreshold(time)) { return true; }
		else { return false; }
	}
	
	// 閾値を返す
	public double getThreshold(double time) {
		double threshold = 1.0;
		updateGameMatrix(); // ゲームの表を更新する
		double target = getExpectedUtilityinFOP() / Math.pow(df, time); // 最終提案フェーズの期待効用（割引効用によって減少する効用値を考慮して上方補正する）
		// 最終提案フェーズの期待効用に基づき，譲歩を行う
		if(df == 1.0){ threshold = target + (1.0 - target) * (1.0 - time); }
		else { threshold = Math.max(1.0 - time / df, target); }		
		
		// デバッグ用
		if(Atlas3.isPrinting){ System.out.println("threshold = " + threshold + ", opponents:"+negotiatingInfo.getOpponents()); }

		return threshold;
	}
			
	// 最終提案フェイズの混合戦略の期待効用
	private double getExpectedUtilityinFOP(){
		double q = getOpponentEES();
		return q*A21+(1-q)*A22;
	}
	
	// 最終提案ゲームにおける最適混合戦略の均衡点での，相手の混合戦略(p,1.0-p)=(強硬戦略を選択する確率，妥協戦略を選択する確率)を導出する
	private double getOpponentEES(){		
		double q = 1.0;
		if((A12 - A22 != 0) && (1.0-(A11-A21)/(A12-A22) != 0)){ q = 1.0 / (1.0 - (A11-A21)/(A12-A22)); }
		if(q < 0.0 || q > 1.0){ q=1.0; }
		return q;
	}
	
	// ゲームの表を更新する
	private void updateGameMatrix(){
		double C;  // 妥協案の推定効用値
		if(negotiatingInfo.getNegotiatorNum() == 2){ C = negotiatingInfo.getBOU(); }
		else { C = negotiatingInfo.getMPBU(); }

		A11 = rv * df;
		A12 = Math.pow(df, TF);
		if(C >= rv) { A21 = C * Math.pow(df,TF); }
		else { A21 = rv * df; }
		A22 = PF * A21 + (1.0-PF) * A12;
	}
}
