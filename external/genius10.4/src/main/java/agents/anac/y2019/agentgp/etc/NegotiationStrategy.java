package agents.anac.y2019.agentgp.etc;

import java.util.ArrayList;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class NegotiationStrategy {
	private AbstractUtilitySpace utilitySpace;

	private double df = 1.0; // 割引係数
	private double rv = 0.0; // 留保価格

	private static double a = 8d;
	private static double s = 0.5;
	private static double w = 0.1;
	private static NormalDistribution nb;
	private ArrayList<Double> x0list;
	private ArrayList<Double> y0list;

	//Aが自分,Bが相手　各種計算などはパワーポイントの説明を参照

	private boolean isPrinting = false;

	public NegotiationStrategy(AbstractUtilitySpace utilitySpace, boolean isPrinting) {
    	nb = new NormalDistribution(0, 0.1);

		this.utilitySpace = utilitySpace;
		this.isPrinting = isPrinting;
		df = utilitySpace.getDiscountFactor();
		rv = utilitySpace.getReservationValue();

		x0list = new ArrayList<Double>(10000);
		y0list = new ArrayList<Double>(10000);

		if(this.isPrinting){ System.out.println("NegotiationStrategy:success"); }
	}

	public void setRV(double newRV){
		rv = newRV;
	}

	public void addBid(double time, double utilityValue) {
		x0list.add(time);
		y0list.add(utilityValue);
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
		if (utilitySpace.discount(rv, time) >= getThreshold(time)) { return true; }
		else { return false; }
	}

	// 割引後の効用値から割引前の効用値を導出する
	public double pureUtility(double discounted_util, double time) {
		return discounted_util / Math.pow(df, time);
	}

	// 閾値を返す
	// timeは0以上1以下
	public double getThreshold(double time) {
		if(time < 0.5)
		//if(time < 0.05)
			return 0.9;
        //double[] x0 = {0d, 1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d, 10d};
        //double[] x0 =
        //{0.38706811451527395 ,0.3206054925561284 ,0.28103018043139005 ,0.24436436083847995 ,0.3445771877283978 ,0.1916617620178223 ,0.10851498138164084 ,0.14185336453254715 ,0.08559170234817393 ,0.17779733823436733 ,0.418914454566757 ,0.3584402105065997 ,0.42008488606407235 ,0.19100081872339525 ,0.031590428414826344 ,0.04030499406302207 ,0.401479922036519 ,0.3660617188556783 ,0.07356412697343317 ,0.3644534922591787 ,0.23259339929628758 ,0.07776247821815369 ,0.2056512338771206 ,0.29111883513606107 ,0.2113101232746351 ,0.05177255209604298 ,0.07996772745779601 ,0.04883214791099755 ,0.13170165644396714 ,0.026885527544211196 ,0.20438879829276113 ,0.3985902442092543 ,0.06459996360985837 ,0.10835947390326844 ,0.3882420720097844 ,0.4248451139206499 ,0.459510096971506 ,0.12771475963851137 ,0.34753640913107364 ,0.16806010397251636 ,0.4324935074362608 ,0.40288573838467034 ,0.015800608845888242 ,0.3904905152422249 ,0.12668070830772887 ,0.06507005712215841 ,0.3082148526114499 ,0.2108467685322144 ,0.16415254522588912 ,0.11103689994797744 ,0.06754357754521073 ,0.4942182273579348 ,0.4791624688216025 ,0.4702471825912632 ,0.01316764413161886 ,0.3926145037660956 ,0.37713371770240944 ,0.4968927584512328 ,0.38560739916143455 ,0.4199562499040234 ,0.2834817654284025 ,0.3889643713124304 ,0.1138823113683583 ,0.2096256732162426 ,0.11191649637646434 ,0.45270385115385037 ,0.4128654562345271 ,0.47156219269239613 ,0.13081991509309004 ,0.46916165224139766 ,0.2286497862943937 ,0.47877473174822255 ,0.49336271354309214 ,0.31060516860021326 ,0.022285551411652837 ,0.41794303123711696 ,0.07771612503113112 ,0.40796451817774004 ,0.36710588973449004 ,0.10306776486868907 ,0.21780230433091774 ,0.13925922897833742 ,0.11849176638730208 ,0.4667890057172684 ,0.4930224055342757 ,0.333805279554561 ,0.08832958042831213 ,0.17550728514664726 ,0.05391632283899184 ,0.3000088304504989 ,0.07943328666622745 ,0.4419313191573395 ,0.013972144714187007 ,0.01945247983056253 ,0.46756204818124786 ,0.2518159147391108 ,0.320818889962676 ,0.4627323074039614 ,0.15170377565015525 ,0.4967412812878962 ,0.06916248007812886 ,0.16610851222887596 ,0.09247770142141565 ,0.24532208300734398 ,0.16646326999402594 ,0.1762834816888213 ,0.2858867121449584 ,0.23260578294578216 ,0.3722238402071553 ,0.014792425572133061 ,0.23742130428338554 ,0.12278289006353676 ,0.38484262141662534 ,0.10830385368423895 ,0.09020850873656738 ,0.15427035576474007 ,0.41487654924195383 ,0.360083563213973 ,0.052272766665469705 ,0.3359767081705856 ,0.061697728452053346 ,0.12471216043378103 ,0.37042184434571346 ,0.43659567152346485 ,0.03790532917878353 ,0.22284811462797888 ,0.11854294167324309 ,0.18232348036688273 ,0.4041021993260984 ,0.1759997496823793 ,0.14545081563348128 ,0.027894391887294423 ,0.467398853248195 ,0.47986435346568296 ,0.1281934148659663 ,0.33178790254924656 ,0.29426390666923946 ,0.445604862963029 ,0.04602823170103543 ,0.459355065451731 ,0.3009219003143079 ,0.4071571598236535 ,0.3418977057166697 ,0.35148107725021965 ,0.1673228045351277 ,0.1287178223314614 ,0.4761046174114128 ,0.33897591931578547 ,0.1654233775967866 ,0.2248049735940163 ,0.22788872615730882 ,0.053220758998040074 ,0.10676155827323297 ,0.11853757868044645 ,0.32713951986122597 ,0.348809186310242 ,0.04122896061784709 ,0.0718776922688823 ,0.27169385517763184 ,0.03794700812822954 ,0.3588082591692302 ,0.4916296121167893 ,0.05305935392735939 ,0.2715465194205913 ,0.38957825440180865 ,0.17765482130523585 ,0.1978551289683143 ,0.13101853947911252 ,0.24485388356598897 ,0.1240932384158982 ,0.38843707100587394 ,0.49068430263486307 ,0.1457089986919488 ,0.2843480943575814 ,0.3920925786954471 ,0.04261569176755692 ,0.12417375923078883 ,0.04857089137424658 ,0.36920697470656444 ,0.15697760117452814 ,0.398441787639602 ,0.35231202622204144 ,0.43688061118699195 ,0.4438067138305715 ,0.1407618809296567 ,0.24579765429447759 ,0.03173614000002689 ,0.3030138760799205 ,0.45409172825971444 ,0.49945951994395416 ,0.004101639913858146 ,0.18136956230004592 ,0.3699166211178163 ,0.20820495222800894 ,0.2554398876945577 ,0.07920182850557544 ,0.12220881964631658 ,0.2682570566996679 ,0.16519522889117583 ,0.3686244635239205 ,0.19755137979466814 ,0.26365168484672974 ,0.14249265531773475 ,0.06507203687039237 ,0.040386653178833276 ,0.4190690603736449 ,0.42646825370491986 ,0.4865731602780412 ,0.24467884424017067 ,0.3085957231323109 ,0.17126538786466827 ,0.40487457754211836 ,0.33869715586309374 ,0.3433024081792852 ,0.4430832264455144 ,0.21104582901449137 ,0.06144782628742346 ,0.36137213498656 ,0.3503338770796483 ,0.4054919726288751 ,0.06748033046201324 ,0.1656581132836571 ,0.33014819923422833 ,0.4571961039551598 ,0.15956456967112026 ,0.17415088826304637 ,0.0968700906866024 ,0.06953005114656668 ,0.2334597021839228 ,0.4899493022998252 ,0.4719285376450565 ,0.1772743988273981 ,0.012143709231353106 ,0.00032168563786449944 ,0.21463000955003908 ,0.46627245060378986 ,0.06253230359740597 ,0.32995356422667854 ,0.03779506744792516 ,0.0577925496452259 ,0.46735906893956614 ,0.04029619790110606 ,0.1535303800695017 ,0.43820280876630296 ,0.12684627443131757 ,0.28354009924843276 ,0.29957218212790726 ,0.10160211344364684 ,0.1727769699306721 ,0.07523357452040047 ,0.11904229074320577 ,0.31271538725586817 ,0.12564699471054275 ,0.2923155514274439 ,0.41396715829775754 ,0.29074143741171 ,0.39256331252214915 ,0.2930731822810519 ,0.4715083622684976 ,0.26173171567575293 ,0.47918139048159636 ,0.40489372580875477 ,0.24899696952549072 ,0.042811163604270674 ,0.36048124457185227 ,0.31714259455584315 ,0.35920109626549745 ,0.2885427735560154 ,0.35964687414985475 ,0.4685818453855714 ,0.2898073590611159 ,0.433561433827509 ,0.21362387739246386 ,0.40372971155216403 ,0.20724234140062503 ,0.04721877610281505 ,0.08351204671039547 ,0.2090204432955317 ,0.08964617996652502 ,0.19907242001179903 ,0.006231989627947909 ,0.01813160981651374 ,0.34674960300248564 ,0.20262255241249588 ,0.2384698659683449 ,0.45647408094544684 ,0.027212778950331085 ,0.2672724480271924 ,0.46810361103857495 ,0.19460134635464632 ,0.07385878239902582 ,0.10805316518256802 ,0.23838014494874543 ,0.008272562219645618 ,0.15802807219292242 ,0.3711291654556587 ,0.14253855483228206 ,0.24827793152364847 ,0.10012353746626901 ,0.21717649251053728};
		double[] x0 = new double[x0list.size()];
		for(int i=0; i < x0list.size(); i++)
			x0[i] = x0list.get(i);
		double[] y0 = new double[y0list.size()];
		for(int i=0; i < y0list.size(); i++)
			y0[i] = y0list.get(i);
		//double[] y0 = (Double[]) y0list.toArray();
        //double[][] y0;
        //y0 = new double[1][x0.length];
        //for(int i=0; i < x0.length; i++) {
        //	y0[0][i] = y(x0[i]) + nb.sample();
        //}
        RealMatrix k00 = RBFMatrix(x0, x0.length, x0, x0.length, false);
		RealMatrix k00_1 = new LUDecomposition(k00).getSolver().getInverse();

		/*
		Double[] x1 = new Double[1000-x0.length];
        x1[0] = time;
        double sub = (1d - (x1[0])) / (double)(x1.length-1);
        for(int i=1; i < x1.length; i++) {
        	x1[i] = x1[i-1] + sub;
        }
        x1[x1.length-1] = 1d;
        */
		double[] x1 = new double[100];
        x1[0] = time;
        double x1max = 1d;
        if(time + 0.1 < 1)
        	x1max = time + 0.1;
        double sub = (x1max - (x1[0])) / (double)(x1.length-1);
        for(int i=1; i < x1.length; i++) {
        	x1[i] = x1[i-1] + sub;
        }
        x1[x1.length-1] = x1max;

        RealMatrix k01 = RBFMatrix(x0, x0.length, x1, x1.length, true);
        RealMatrix k10 = k01.transpose().copy();
        RealMatrix k11 = RBFMatrix(x1, x1.length, x1, x1.length, false);

		double[][] y0d = new double[1][y0.length];
		for(int i=0; i < y0.length; i++)
			y0d[0][i] = y0[i];
        RealMatrix mu = k10.multiply(k00_1.multiply(MatrixUtils.createRealMatrix(y0d).transpose()));
        //System.out.println(mu.getRowDimension() + ", " + mu.getColumnDimension());
        //showMatrix(mu);
        //writeMatrix1d(mu);
        RealMatrix sigma = k11.subtract(k10.multiply(k00_1.multiply(k01)));
        //System.out.println(sigma.getRowDimension() + ", " + sigma.getColumnDimension());
        //showMatrix(sigma);
        //writeMatrix(sigma);
        //showDiagonal(sigma);
        double[] mud = mu.getColumn(0);
        System.out.println(x0.length + ":" + mud.length);
        double max = getMaxValue(mud, sigma.getData());
        System.out.println("max:" + max);
        if(max < 0.3)
        	return 0.9;
		return max;
	}

	private void showDiagonal(RealMatrix m) {
		double[][] md = m.getData();
		for(int i=0; i < md.length; i++)
			System.out.print(md[i][i] + ", ");
		System.out.println();
	}

	private double getMaxValue(double[] mean, double[][] dist) {
		double max = -1000;
//		for(int i=start_index; i < mean.length; i++) {
//        	System.out.println(i + ":" + (mean[i]));
//        	if(mean[i] > 1)
//        		return 1;
//        	else if(max < mean[i])
//				max = mean[i];
//		}
		for(int i=0; i < mean.length; i++) {
        	//System.out.println(i + ":" + (mean[i]+Math.sqrt(dist[i][i])));
        	if(mean[i]+Math.sqrt(dist[i][i]) > 1) {
        		max = 1d;
        		break;
        	}
        	else if(max < mean[i]+Math.sqrt(dist[i][i]))
				max = mean[i]+Math.sqrt(dist[i][i]);
		}
		return max;
	}

    private static double RBFkernel(double x1, double x2) {
    	return Math.pow(a, 2.0) * Math.exp(-Math.pow((x1-x2)/s, 2));
    }

    private static RealMatrix RBFMatrix(double[] x0, int size1, double[] x02, int size2, boolean indexing) {
    	double[][] mat;
    	if(indexing==false) {
			mat = new double[size2][size1];
			for(int i=0; i < size2; i++) {
				for(int j=0; j < size1; j++) {
					if(i != j)
						mat[i][j] = RBFkernel(x0[j], x02[i]);
					else
						mat[i][j] = RBFkernel(x0[j], x02[i]) + w;
//					mat[i][j] = vec1[j] + vec2[i];
				}
			}
			return MatrixUtils.createRealMatrix(mat);
    	}
    	else {
			mat = new double[size1][size2];
			for(int i=0; i < size1; i++) {
				for(int j=0; j < size2; j++) {
					if(i != j)
						mat[i][j] = RBFkernel(x0[i], x02[j]);
					else
						mat[i][j] = RBFkernel(x0[i], x02[j]) + w;
//					mat[i][j] = vec1[i] + vec2[j];
				}
			}
			return MatrixUtils.createRealMatrix(mat);
    	}
    }
    /**
     * 行列の表示
     * @param m
     */
    private static void showMatrix(RealMatrix m) {
        System.out.println("----------------");
        for (int i = 0; i < m.getRowDimension(); i++) {
            System.out.print("{");
            for (int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.getEntry(i, j) + ", ");
            }
            System.out.println("}");
        }
    }

    private static void writeMatrix1d(RealMatrix m) {
        try{
			File file = new File("/Users/fukui/Desktop/test.txt");
			if (checkBeforeWritefile(file)){
				BufferedWriter bw = new BufferedWriter(new FileWriter(file));
				bw.write("{\"mu\":");
				bw.write("[");
				for (int j = 0; j < m.getRowDimension(); j++) {
					if(j != m.getRowDimension()-1)
						bw.write(m.getEntry(j, 0) + ", ");
					else
						bw.write(m.getEntry(j, 0) + " ");
				}
				bw.write("],");
				bw.newLine();

				bw.close();
            }else{
				System.out.println("ファイルに書き込めません");
            }
        }catch(IOException e){
            System.out.println(e);
        }
    }

    private static void writeMatrix(RealMatrix m) {
        try{
			File file = new File("/Users/fukui/Desktop/test.txt");
			if (checkBeforeWritefile(file)){
				BufferedWriter bw = new BufferedWriter(new FileWriter(file, true));
				bw.write("\"sigma\":");
				bw.write("[");
				for (int i = 0; i < m.getRowDimension(); i++) {
					bw.write("[");
					for (int j = 0; j < m.getColumnDimension(); j++) {
						if(j != m.getColumnDimension()-1)
							bw.write(m.getEntry(i, j) + ", ");
						else
							bw.write(m.getEntry(i, j) + " ");
					}
					if(i != m.getRowDimension()-1)
						bw.write("], ");
					else
						bw.write("]");
				}
				bw.write("]}");
				bw.newLine();

				bw.close();
            }else{
				System.out.println("ファイルに書き込めません");
            }
        }catch(IOException e){
            System.out.println(e);
        }

    }

    private static void write(String m) {
        try{
			File file = new File("/Users/fukui/Desktop/test.txt");
			if (checkBeforeWritefile(file)){
				FileWriter fw = new FileWriter(file, true);
				PrintWriter pw = new PrintWriter(new BufferedWriter(fw));
				pw.println(m);
				pw.close();
            }else{
				System.out.println("ファイルに書き込めません");
            }
        }catch(IOException e){
            System.out.println(e);
        }

    }

    private static boolean checkBeforeWritefile(File file){
        if (file.exists()){
			if (file.isFile() && file.canWrite()){
				return true;
			}
        }
        return false;
    }
}

