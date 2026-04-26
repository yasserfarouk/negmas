package agents.anac.y2015.fairy;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class strategy {
	private AdditiveUtilitySpace utilitySpace;
	private negotiatingInfo negotiatingInfo;
	//inamoto
	public double minThreshold = 0.8;
	
	private double df = 0.0; // 髯ｷ謇假ｽｽ�ｲ髯滄��ｩ繧托ｽｽ�ｿ郢ｧ閧ｲ�ｽ
	private double rv = 0.0; // 鬨ｾ�｡陷ｷ�ｩ�ｽ�ｿ隴乗��ｽ�ｾ�ｽ�｡髫ｴ�ｬ�ｽ�ｼ

	public strategy(AdditiveUtilitySpace utilitySpace, negotiatingInfo negotiatingInfo) {		
		this.utilitySpace = utilitySpace;
		this.negotiatingInfo = negotiatingInfo;
		df = utilitySpace.getDiscountFactor();
		rv = utilitySpace.getReservationValue();
	}
	
	// 髯ｷ�ｿ隲､諛ｶ�ｽ�ｮ�ｽ�ｹ髯具ｽｻ�ｽ�､髯橸ｽｳ�ｽ�ｽ
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			double offeredBidUtil = utilitySpace.getUtility(offeredBid);
			
			if(offeredBidUtil >= getThreshold(time)){ return true; }
			else{ return false; }
		} catch (Exception e) {
			System.out.println("髯ｷ�ｿ隲､諛ｶ�ｽ�ｮ�ｽ�ｹ髯具ｽｻ�ｽ�､髯橸ｽｳ陞｢�ｹ遶頑･｢譽費ｿｽ�ｱ髫ｰ�ｨ陷会ｽｱ�ｽ�ｽ驍ｵ�ｺ�ｽ�ｾ驍ｵ�ｺ陷会ｽｱ隨ｳ�ｽ");
			e.printStackTrace();
			return false;
		}
	}
	
	// 髣費ｿｽ�ｽ�､髮九ｊ�ｽ�ｽ�ｵ郢ｧ�ｽ�ｽ�ｺ�ｽ�ｽ隲｢蟷�･懶ｿｽ�ｽ
	public boolean selectEndNegotiation(double time) {
		if(rv * Math.pow(df,time) > getThreshold(time) ) return true;
		
		return false;
	}
	
	// 鬯ｮ�｢�ｽ�ｾ髯区ｻゑｽｽ�､驛｢�ｧ陞ｳ螟ｲ�ｽ�ｿ隴∫ｵｶ�ｽ
	public double getThreshold(double time) {

		
		double a = minThreshold; //隴幢ｿｽ闖ｴ蠑ｱ縲堤ｹｧ繧�ｲ玖ｫ｢荵怜�邵ｺ�ｫ陟募干笳�ｸｺ�ｽ��邵ｺ髦ｪ�櫁屐�､(陷亥沺辟皮ｸｺ�ｮ陟托ｽｷ雎鯉ｿｽ)
//		double threshold = 1.0 - (1.0 - a) * time;
		//boulware鬚ｨ縺ｫ險よｭ｣
		double threshold = 1.0 - (1.0 - a ) * Math.pow(time, (1.0/0.5));
//隰厄ｿｽ�ｮ螢ｹ��ｹｧ蠕娯�ｻ邵ｺ�ｽ�玖屐�､邵ｺ�ｫ陷ｷ莉｣ﾂｰ邵ｺ�｣邵ｺ�ｦ騾ｶ�ｴ驍ｱ螟ょ飭邵ｺ�ｫ陞溽甥陌夂ｸｺ蜷ｶ�狗ｸｺ蜉ｱ窶ｳ邵ｺ�ｽ�ｽ�､邵ｺ�ｮ陞ｳ貅ｽ讓溽ｸｲ�ｽ
		
//		System.out.println("test round:"+time);
//		System.out.println("test time:"+(double)time/180);
		/* 髣費ｿｽ�ｽ�､髮九ｉ蛻､陝具ｽｶ鬨ｾ�｡�ｽ�･驍ｵ�ｺ�ｽ�ｫ髯憺屮�ｽ�ｺ驍ｵ�ｺ�ｽ�･驍ｵ�ｺ髯ｦ�｡hreshold(t)驛｢�ｧ陞ｳ螟ｲ�ｽ�ｨ�ｽ�ｭ鬮ｫ�ｪ陋ｹ�ｻ隨假ｿｽ�ｹ�ｧ�ｽ�ｽ */
		/* negotiatingInfo驍ｵ�ｺ闕ｵ譎｢�ｽ闃ｽ�ｬ�ｽ陷郁肩�ｽ�｡闔��･�ｽ�ｱ�ｽ�･髮趣ｿｽ�ｽ�ｴ驍ｵ�ｺ�ｽ�ｮ鬩搾ｽｨ�ｽ�ｱ鬮ｫ�ｪ陜捺ｺ佩鈴劈�｣�ｽ�ｱ驛｢�ｧ髮区ｧｫ蠕宣辧蜍溷ｹｲ邵ｲ蝣､�ｸ�ｺ鬮ｦ�ｪ�ｽ迢暦ｽｸ�ｺ�ｽ�ｮ驍ｵ�ｺ�ｽ�ｧ髣厄ｽｴ�ｽ�ｿ驍ｵ�ｺ�ｽ�｣驍ｵ�ｺ�ｽ�ｦ驛｢�ｧ郢ｧ鬆托ｿｽ驍ｵ�ｺ�ｽ�ｽ */
		// 髣懃§�ｽ�ｽ�ｼ�ｽ�ｽ
		ArrayList<Object> opponents =  negotiatingInfo.getOpponents();
		//闔臥ｴ具ｽｺ�ｺ邵ｺ謔溽ｲ玖ｫ｢荳奇ｽ帝ｨｾ�ｲ郢ｧ竏壺�ｻ邵ｺ�ｽ�狗ｸｺ�ｨ邵ｺ髦ｪ竊鍋ｸｺ�ｯ髢ｾ�ｪ髴�ｽｫ郢ｧ繧会ｽｩ閧ｴ�･�ｵ騾ｧ�ｽ竊楢ｱｬ竏夲ｼ�ｹｧ蠕鯉ｽ狗ｸｲ�ｽ(陟墓｢ｧ辟皮ｸｺ�ｮ陟托ｽｱ雎鯉ｿｽ)
		int acceptNum = 0;
		for(Object sender:opponents){	
			if(negotiatingInfo.getOpponentsBool(sender))
				acceptNum += 1;
		}
		int negotiatorNum = negotiatingInfo.getNegotiatorNum()-1;//髢ｾ�ｪ陋ｻ�ｽ�帝ｫｯ�､邵ｺ�ｽ笳�滋�､雋り歓�ｽ�ｽ�ｽ隰ｨ�ｰ
		
//		System.out.println("test:"+(double)acceptNum/negotiatorNum);
		
		threshold = threshold - (threshold - minThreshold)*(double)acceptNum/negotiatorNum;
		//陷ｻ�ｨ郢ｧ鄙ｫ窶ｲ陷ｷ蝓湲咲ｸｺ蜉ｱ窶ｻ邵ｺ�ｽ笳�ｹｧ迚咏ｲ玖ｫ｢荳奇ｼ�邵ｺ�ｦ邵ｺ�ｽ�玖滋�ｺ隰ｨ�ｰ邵ｺ�ｮ驕抵ｽｺ驍�ｿｽ�ｽ邵ｺ�ｽ邵ｺ蜿ｰ�ｺ�､雋ょｳｨ�ｽ邵ｺ蜉ｱ窶ｳ邵ｺ�ｽ�ｽ�､郢ｧ蜑�ｽｸ荵敖｣郢ｧ荵晢ｿｽ繧�螺邵ｺ�ｽ邵ｺ蜉ｱ�ｽ竏ｵ諤呵抄蠑ｱ縲堤ｹｧ�ｮinThreshold邵ｺ�ｾ邵ｺ�ｧ
		//髫ｪ閧ｲ�ｮ蜉ｱ��邵ｺ貅假ｼ�邵ｺ髦ｪ�櫁屐�､-(髫ｪ閧ｲ�ｮ蜉ｱ��邵ｺ貅假ｼ�邵ｺ髦ｪ�櫁屐�､-隴幢ｿｽ闖ｴ蠑ｱ�ｽ邵ｺ蜉ｱ窶ｳ邵ｺ�ｽ�ｽ�､)*驕抵ｽｺ驍�ｿｽ
		
		negotiatingInfo.clearOpponentsBool();//闔��､雋り�諞ｾ隲ｷ荵晢ｿｽ郢ｧ�ｯ郢晢ｽｪ郢ｧ�｢
/*螯･蜊斐＠縺吶℃縺ｪ縺�ｈ縺�↓菫ｮ豁｣縺吶ｋ縺ｫ縺ｯ
 *逶ｸ謇九′蜊泌鴨逧�°髱槫鵠蜉帷噪縺九ｒ蛻､譁ｭ縺吶ｋ蠢�ｦ√′縺ゅｋ縲�
 * 蜊泌鴨逧�°縺､雉帛酔竊偵％縺｡繧峨�蠑ｷ豌励↓蜃ｺ縺ｦ繧ゅｈ縺�
 * 髱槫鵠蜉帷噪縺九▽雉帛酔竊偵％縺｡繧峨�蠑ｱ豌励↓蜃ｺ縺溘⊇縺�′繧医＞
 * */
		//割引雇用の適用
		threshold = threshold * Math.pow(df,time);
		//最低の値は守る。です。
		if(threshold < minThreshold) threshold = minThreshold;
		
//		System.out.println("threshold:"+threshold);
		return threshold;
	}
}
