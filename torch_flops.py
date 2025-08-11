from fvcore.nn import FlopCountAnalysis, flop_count_table


model = "선언할 모델"
input_img = torch.ones("shape 입력")

flops = FlopCountAnalysis(Teacher_model, input_img)

print(flops.total()) # kb단위로 모델전체 FLOPs 출력해줌
print(flop_count_table(flops)) # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌