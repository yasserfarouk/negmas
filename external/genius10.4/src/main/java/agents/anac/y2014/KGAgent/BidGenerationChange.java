package agents.anac.y2014.KGAgent;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import agents.anac.y2014.kGA_gent.library_genetic.Gene;
import agents.anac.y2014.kGA_gent.library_genetic.GenerationChange;

public class BidGenerationChange implements GenerationChange{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–

	}


	int size = 50;

	int end = 40;

	double line = 2.0;

	static Random rnd=null;

	int gen = 0;

	public BidGenerationChange() {

		//System.out.println("Call BidGenerationChange Instance");
		if(rnd==null){
			rnd = new Random();
		}
	}
	public BidGenerationChange(int size) {

		this.size  =size;
		//System.out.println("Call BidGenerationChange Instance");
		if(rnd==null){
			rnd = new Random();
		}
	}

	public BidGenerationChange(int size,int gen) {

		end=gen;
		this.size  =size;
		//System.out.println("Call BidGenerationChange Instance");
		if(rnd==null){
			rnd = new Random();
		}
	}

	/*
	 * ãƒˆãƒ¼ãƒŠãƒ¡ãƒ³ãƒˆæˆ¦ç•¥ ãƒ©ãƒ³ãƒ€ãƒ ã�«å€‹ä½“é�¸æŠžè‚¢ã��ã�®1ç•ªã�¨2ç•ªã�§äº¤å�‰ã‚’è¡Œã�†
	 */
	static int tornament = 8;


	@Override
	public List<Gene> Generation(List<Gene> list) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–

		//System.out.println("call Generation");

		ArrayList<Gene> ret = new ArrayList<Gene>(size);

		int f=size-2,s=size-1,buf;

		gen ++;

		MyBidGene bufBidGene = (MyBidGene)list.get(0);

		ret.add(new MyBidGene(bufBidGene));

		while (ret.size() < size) {

			f=size-2;
			s=size-1;


			for (int i = 0; i < tornament; i++) {

				buf = rnd.nextInt(size);

				s = Math.min(buf, s);
				f = Math.min(s, f);

			}
			Gene b = list.get(s).Cros(list.get(f));
			b.Mutate();
			ret.add(b);
		}

		//System.out.println("end Generation");


		return ret;
	}

	@Override
	public List<Gene> StartGeneration(Gene gene) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–
		List<Gene> ret = new ArrayList<Gene>(size);
		while (ret.size()<size) {
			ret.add(new MyBidGene());
		}
		return ret;
	}

	public List<Gene> StartGeneration() {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–
	//	System.out.println("Calll StartGeneration");
		List<Gene> ret = new ArrayList<Gene>(size);
		while (ret.size()<size) {
			ret.add(new MyBidGene());
		}
		return ret;
	}

	@Override
	public boolean End(List<Gene> list) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–
		if(gen > end){
			return true;
		}
		return false;
	}

}
