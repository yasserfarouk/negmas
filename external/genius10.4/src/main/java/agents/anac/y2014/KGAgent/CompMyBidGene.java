package agents.anac.y2014.KGAgent;

import agents.anac.y2014.kGA_gent.library_genetic.CompGene;
import agents.anac.y2014.kGA_gent.library_genetic.Gene;

public class CompMyBidGene extends CompGene{


	int type = 0;
	public CompMyBidGene(int type) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿ãƒ¼ãƒ»ã‚¹ã‚¿ãƒ–
		this.type = type;
	}
	public CompMyBidGene(){
	}

	@Override
	public int compare(Gene o1, Gene o2) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–

		MyBidGene b1 = (MyBidGene)o1;
		MyBidGene b2 = (MyBidGene)o2;
		double d = b1.GetValue(type) - b2.GetValue(type);
		if(d > 0){
			return 1;
		}else if (d<0) {
			return -1;
		}else{
			return 0;
		}
	}

}
