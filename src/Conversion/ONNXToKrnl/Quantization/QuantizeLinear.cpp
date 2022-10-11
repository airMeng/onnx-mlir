/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- QuantizeLinear.cpp - Lowering QuantizeLinear Ops ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX QuantizeLinear Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXQuantizeLinearOpLowering
    : public ConversionPattern {
  ONNXQuantizeLinearOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXQuantizeLinearOp::getOperationName(), 1,
            ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // quantizelinear(x, scale, zero_point) =
    //      int8_t(x / scale) + zero_point  if dtype(zero_point) == int8_t
    //      uint8_t(x / scale) + zero_point  if dtype(zero_point) == uint8_t
    ONNXQuantizeLinearOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    Value operand = operandAdaptor.x();
    Value scale = operandAdaptor.y_scale();
    Value zeroPoint = operandAdaptor.y_zero_point();

    // Insert an allocation and deallocation for the result of this operation.
    bool insertDealloc = checkInsertDealloc(op);

    Value alloc =
        (hasAllConstantDimensions(memRefType))
            ? insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc)
            : insertAllocAndDealloc(
                  memRefType, loc, rewriter, insertDealloc, {operand});

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
    // In case of N, C is assumed to be 1.
    // Shapes of scale and zero_point must be 1 or C.
    // Computation of QuantizeLinear is done as if scale and
    // zero_point are reshaped to Cx1x1x...x1.

    // rank
    int64_t rank = memRefType.getRank();
    Type elementType = memRefType.getElementType();

    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, rank);

    // Create a KrnlIterateOp along C dimension.
    // This will be the outer-most loop in order to re-use scale and zero_points

    SmallVector<Value, 1> loopCIVs;
    if (rank > 1) {
      // TODO use new KrnlDialectBuilder.
      krnl::KrnlIterateOperandPack cPack(rewriter, originalLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      KrnlIterateOp cIterateOp = create.krnl.iterate(cPack);
      Block &cIterationBlock = cIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else
      loopCIVs.emplace_back(create.math.constantIndex(0));

    Value scaleVal = create.krnl.load(scale, loopCIVs);
    Value zeroPointVal = create.krnl.load(zeroPoint, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops;
    for (size_t i = 0; i < axes.size(); ++i)
      packLoops.emplace_back(originalLoops[axes[i]]);

    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, packLoops);
    for (size_t i = 0; i < axes.size(); ++i)
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    SmallVector<Value, 4> loopIVs;
    auto args = iterationBlock.getArguments();
    if (args.size() > 1) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
      for (unsigned int i = 1; i < args.size(); ++i)
        loopIVs.emplace_back(args[i]);
    } else if (rank == 2) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
    } else
      loopIVs.emplace_back(args[0]);

    Value xVal = create.krnl.load(operand, loopIVs);
    Value normVal = create.math.div(xVal, scaleVal);
    Value floatZeroPointVal = create.math.cast(rewriter.getF32Type(), zeroPointVal);
    Value shiftedVal = create.math.add(normVal, floatZeroPointVal);
    Type dstType = operandAdaptor.y_zero_point().getType().cast<ShapedType>().getElementType();
    int64_t dstWidth = dstType.getIntOrFloatBitWidth();
    if (dstType.isUnsignedInteger()) {
      Value unsignedDstVal = create.math.cast(rewriter.getIntegerType(dstWidth, false), shiftedVal);
      create.krnl.store(unsignedDstVal, alloc, loopIVs);
   } else {
      Value signedDstVal = create.math.cast(rewriter.getIntegerType(dstWidth, true), shiftedVal);
      create.krnl.store(signedDstVal, alloc, loopIVs);
  }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXQuantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXQuantizeLinearOpLowering>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
