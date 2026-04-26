package agents.anac.y2019.minf.etc;

import java.util.Random;

public class NegotiationInfo {
    private double alpha = 1.0D;

    private double OpponentAve = 0.0D;
    private double OpponentSum = 0.0D;
    private double OpponentPowSum = 0.0D;
    private double OpponentVar = 0.0D;
    private double OpponentStdDev = 0.0D;
    private int OpponentNum = 0;

    private double OpponentOwnAve = 0.0D;
    private double OpponentOwnSum = 0.0D;
    private double OpponentOwnPowSum = 0.0D;
    private double OpponentOwnVar = 0.0D;
    private double OpponentOwnStdDev = 0.0D;
    private double OpponentMinUtil = 1.0D;
    private int OpponentOwnNum = 0;


    private double MyAve = 0.0D;
    private double MySum = 0.0D;
    private double MyPowSum = 0.0D;
    private double MyVar = 0.0D;
    private double MyStdDev = 0.0D;
    private int MyNum = 0;

    private boolean isDiscounted = false;
    private double df = 1.0D;
    private double rv = 0.0D;

    private boolean isFirst = false;

    private Random rand = new Random();

    public NegotiationInfo(){
    }

    public NegotiationInfo(double df, double rv){
        this.df = df;
        this.rv = rv;
        this.isDiscounted = df != 1.0;
    }

    public void updateInfo(double lastOpponentBidUtil){
        this.OpponentNum += 1;
        this.OpponentSum += lastOpponentBidUtil;
        this.OpponentPowSum += Math.pow(lastOpponentBidUtil, 2.0D);
        this.OpponentAve = this.OpponentSum / (double)OpponentNum;
        this.OpponentVar = this.OpponentPowSum / (double)OpponentNum - Math.pow(this.OpponentAve, 2.0D);
        if (this.OpponentVar < 1.0E-8) this.OpponentVar = 0.0D;
        this.OpponentStdDev = Math.sqrt(this.OpponentVar);
    }

    public void updateOwnInfo(double lastOpponentOwnBidUtil){
        this.OpponentOwnNum += 1;
        this.OpponentOwnSum += lastOpponentOwnBidUtil;
        this.OpponentOwnPowSum += Math.pow(lastOpponentOwnBidUtil, 2.0D);
        this.OpponentOwnAve = this.OpponentOwnSum / (double)OpponentOwnNum;
        this.OpponentOwnVar = this.OpponentOwnPowSum / (double)OpponentOwnNum - Math.pow(this.OpponentOwnAve, 2.0D);
        if (this.OpponentOwnVar < 1.0E-8) this.OpponentOwnVar = 0.0D;
        this.OpponentOwnStdDev = Math.sqrt(this.OpponentOwnVar);
        this.OpponentMinUtil = Math.min(this.OpponentMinUtil, lastOpponentOwnBidUtil);
    }

    public void updateMyInfo(double lastMyBidUtil){
        this.MyNum += 1;
        this.MySum += lastMyBidUtil;
        this.MyPowSum += Math.pow(lastMyBidUtil, 2.0D);
        this.MyAve = this.MySum / (double)MyNum;
        this.MyVar = this.MyPowSum / (double)MyNum - Math.pow(this.MyAve, 2.0D);
        if (this.MyVar < 1.0E-8) this.MyVar = 0.0D;
        this.MyStdDev = Math.sqrt(this.MyVar);
    }

    public double getRandomThreshold(double t){
        double threshold = getThreshold(t);

        return threshold + (isDiscounted ? 0.0D : rand.nextDouble() * (1.0D - threshold));
    }

    public double getThreshold(double t){
        double lower = this.LowerLimitThreshold(t);
        double stastical = this.isDiscounted ? this.discount_target(t) : this.target(t);

        if (calcAlpha() >= 0.3D && this.OpponentVar != 0.0D) {
            return Math.max(lower, stastical);
        } else {
            return 1.0D;
        }
    }

    private double LowerLimitThreshold(double t){
        double ret = this.rv * 0.2D + (this.isDiscounted ? 0.3D : 0.55D);
        ret = Math.max(ret, Math.min(1.0D, emax()));

        if (this.rv == 0.0D && t > 0.99D) { ret *= 0.85D; }
        return ret;
    }

    private double discount_target(double t){
        return Math.min(0.95D, 1.0D - (1.0D - this.df) * Math.log1p(t * 1.718281828459045D));
    }

    private double target(double t){
        this.alpha = Math.max(0.1D, 3.5D + this.rv - this.calcAlpha());
        return 1 - (1 - emax()) * Math.pow(t, this.alpha);
    }

    private double emax(){
        return this.OpponentAve + (1 - this.OpponentAve) * d();
    }

    private double d(){
        double ave = this.OpponentAve;
        double StdDev = this.OpponentStdDev;
        if(ave <= 0.0D || ave >= 1.0D){
            return Math.sqrt(12.0D) * StdDev;
        }else {
            return Math.sqrt(3.0D / (ave - ave * ave)) * StdDev;
        }
    }

    private double calcAlpha() {
        return this.calcCooperativeness() - this.calcAssertiveness();
    }

    private double calcAssertiveness() {
        return this.MyVar - this.OpponentVar;
    }

    private double calcCooperativeness() {
        return this.MyAve - this.OpponentAve;
    }

    public double getAlpha(){ return calcAlpha(); }

    public void setFirst(boolean bool) { this.isFirst = bool; }

    public double getOpponentOwnAve() { return this.OpponentOwnAve; }

    public double getOpponentOwnVar() { return this.OpponentOwnVar; }

    public double getOpponentMinUtil() { return this.OpponentMinUtil; }

    public boolean isFirst() { return this.isFirst; }

    public double getAssertiveness() { return this.calcAssertiveness(); }

    public double getCooperativeness() { return this.calcCooperativeness(); }
}
