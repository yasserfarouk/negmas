package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.io.Serializable;

/**
 * A simple tuple class.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class Pair<A, B> implements Serializable {
 
  private A fst;
  private B snd;
 
  public Pair(A fst, B snd) {
    this.fst = fst;
    this.snd = snd;
  }
 
  public A getFirst() { return fst; }
  public B getSecond() { return snd; }
 
  public void setFirst(A v) { fst = v; }
  public void setSecond(B v) { snd = v; }
 
  public String toString() {
    return "Pair[" + fst + "," + snd + "]";
  }
 
  private static boolean equals(Object x, Object y) {
    return (x == null && y == null) || (x != null && x.equals(y));
  }
 
  public boolean equals(Object other) {
     return
      other instanceof Pair &&
      equals(fst, ((Pair<?, ?>)other).fst) &&
      equals(snd, ((Pair<?, ?>)other).snd);
  }
 
  public int hashCode() {
    if (fst == null) return (snd == null) ? 0 : snd.hashCode() + 1;
    else if (snd == null) return fst.hashCode() + 2;
    else return fst.hashCode() * 17 + snd.hashCode();
  }
 
  public static <A,B> Pair<A,B> of(A a, B b) {
    return new Pair<A,B>(a,b);
  }
}