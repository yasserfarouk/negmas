package agents.anac.y2014.kGA_gent.library_genetic;


public class CompGene implements java.util.Comparator<Gene> {

	//               + (x > y)
	// compare x y = 0 (x = y)
	//               - (x < y)
	public int compare(Gene s, Gene t) {
		double x = s.GetValue() - t.GetValue();
		if(x>0){
			return 1;
		}
		if(x<0){
			return -1;
		}
		return 0;
	}
}