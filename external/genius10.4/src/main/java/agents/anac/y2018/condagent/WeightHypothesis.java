package agents.anac.y2018.condagent;


import genius.core.Domain;

public class WeightHypothesis extends Hypothesis {
  double[] fWeight;
  double fAprioriProbability;
  
  public WeightHypothesis(Domain pDomain) { fWeight = new double[pDomain.getIssues().size()]; }
  
  public void setWeight(int index, double value) {
    fWeight[index] = value;
  }
  
  public double getWeight(int index) { return fWeight[index]; }
  
  public String toString() {
    String lResult = "";
    for (double lWeight : fWeight) {
      lResult = lResult + String.format("%1.2f", new Object[] { Double.valueOf(lWeight) }) + ";";
    }
    return lResult;
  }
}
